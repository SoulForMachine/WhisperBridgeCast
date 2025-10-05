import tkinter as tk
from tkinter import font as tkfont

class CaptionsOverlay:
    def __init__(self, root,
                 scroll_speed=800,
                 max_visible_lines=3,
                 line_spacing=5,
                 font_name="Helvetica",
                 font_size=24,
                 padding=10,
                 bottom_margin=50,
                 width_ratio=0.8):

        self.scroll_speed = scroll_speed
        self.max_visible_lines = max_visible_lines
        self.line_spacing = line_spacing
        self.padding = padding
        self.bottom_margin = bottom_margin
        self.font_name = font_name
        self.font_size = font_size
        self.width_ratio = width_ratio

        self.root = root
        self.font = tkfont.Font(family=self.font_name, size=self.font_size)

        # Overlay window
        self.overlay_wnd = tk.Toplevel(root)
        self.overlay_wnd.title("Live Captions")
        self.overlay_wnd.configure(bg="black")
        self.overlay_wnd.attributes("-topmost", True)
        self.overlay_wnd.attributes("-alpha", 0.8)
        self.overlay_wnd.overrideredirect(True)

        self.overlay_wnd.bind("<Button-1>", self.start_move)
        self.overlay_wnd.bind("<B1-Motion>", self.do_move)
        self.start_x = 0
        self.start_y = 0

        # Canvas for rendering text
        self.canvas = tk.Canvas(self.overlay_wnd, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Two buttons centered
        #button_frame = tk.Frame(self.overlay_wnd)
#
        #font_inc = tk.Button(button_frame, text="⇧", fg="white", bg="black", command=self.increase_font_size)
        #font_dec = tk.Button(button_frame, text="⇩", fg="white", bg="black", command=self.decrease_font_size)
        #font_inc.pack(side="left", padx=5)
        #font_dec.pack(side="left", padx=5)
        #button_frame.place(relx=0.5, rely=1.0, anchor="s")

        # Internal state
        self.visual_lines = []  # [{"lines": [canvas_ids], "y": y, "height": pixels}]
        self.pending_text = None  # text waiting to be added
        self.scrolling = False

        # Set fixed window size
        self.set_window_geometry()

    # --- Window movement ---
    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def do_move(self, event):
        x = self.overlay_wnd.winfo_x() + (event.x - self.start_x)
        y = self.overlay_wnd.winfo_y() + (event.y - self.start_y)
        self.overlay_wnd.geometry(f"+{x}+{y}")

    # --- Window sizing and position ---
    def set_window_geometry(self):
        screen_width = self.overlay_wnd.winfo_screenwidth()
        screen_height = self.overlay_wnd.winfo_screenheight()
        width = int(screen_width * self.width_ratio)

        line_height = self.font.metrics("linespace") + self.line_spacing
        height = self.max_visible_lines * line_height + 2 * self.padding

        x = (screen_width - width) // 2
        y = screen_height - height - self.bottom_margin

        self.overlay_wnd.geometry(f"{width}x{height}+{x}+{y}")
        self.width = width
        self.height = height
        self.line_height = line_height
#
#    # --- Font size ---
#    def increase_font_size(self):
#        if self.font_size < 42:
#            bc_pt_x, bc_pt_y = self.get_anchor_pt()
#
#            self.font_size += 2
#            self.label.config(font=("Arial", self.font_size))
#            self.overlay_wnd.update_idletasks()
#
#            self.anchor_wnd_to_pt(bc_pt_x, bc_pt_y)
#
#    def decrease_font_size(self):
#        if self.font_size > 12:
#            bc_pt_x, bc_pt_y = self.get_anchor_pt()
#
#            self.font_size -= 2
#            self.label.config(font=("Arial", self.font_size))
#            self.overlay_wnd.update_idletasks()
#
#            self.anchor_wnd_to_pt(bc_pt_x, bc_pt_y)

    # --- Text wrapping ---
    def wrap_text(self, text):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if self.font.measure(test_line) + 2*self.padding <= self.width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    # --- Public API ---
    def add_text(self, text):
        if self.scrolling:
            # Queue text if scrolling is in progress
            self.pending_text = text
        else:
            self._process_new_text(text)

    def set_last_text(self, text):
        if self.visual_lines:
            # Remove last item's canvas lines
            for cid in self.visual_lines[-1]["lines"]:
                self.canvas.delete(cid)
            # Remove from internal list
            self.visual_lines = self.visual_lines[:-1]
        # Add new text (may scroll old lines if needed)
        self.add_text(text)

    def destroy(self):
        self.overlay_wnd.destroy()

    # --- Internal ---
    def _process_new_text(self, text):
        wrapped_lines = self.wrap_text(text)
        new_item_height = len(wrapped_lines) * self.line_height

        # Compute remaining scroll based on bottom-most old line
        if self.visual_lines:
            old_bottom = max(vl["y"] + vl["height"] for vl in self.visual_lines)
        else:
            old_bottom = 0

        remaining_scroll = max(0, old_bottom + new_item_height - (self.height - self.padding))

        if remaining_scroll > 0:
            # Need to scroll old lines first
            self.scrolling = True
            self._scroll_old_lines(remaining_scroll, new_item_height, wrapped_lines)
        else:
            # Enough space, add immediately
            self._draw_new_lines(wrapped_lines)

    def _draw_new_lines(self, wrapped_lines):
        y = self.height - self.padding - len(wrapped_lines) * self.line_height
        line_ids = []
        for line in wrapped_lines:
            cid = self.canvas.create_text(self.padding, y, anchor="nw",
                                        text=line, fill="white", font=self.font)
            line_ids.append(cid)
            y += self.line_height
        self.visual_lines.append({"lines": line_ids,
                                "y": self.height - self.padding - len(wrapped_lines) * self.line_height,
                                "height": len(wrapped_lines) * self.line_height})
        self._cleanup_lines()

    def _scroll_old_lines(self, remaining_scroll, new_item_height, wrapped_lines):
        if remaining_scroll <= 0:
            self.scrolling = False
            self._draw_new_lines(wrapped_lines)
            if self.pending_text:
                text = self.pending_text
                self.pending_text = None
                self.add_text(text)
            return

        frame_delay = 20  # ms
        # Compute scroll step in pixels per frame
        scroll_step = max(1, self.scroll_speed * frame_delay / 1000)

        # Scroll all old lines up
        for vl in self.visual_lines:
            for cid in vl["lines"]:
                self.canvas.move(cid, 0, -scroll_step)
            vl["y"] -= scroll_step

        remaining_scroll -= scroll_step

        if remaining_scroll <= 0:
            # final adjustment to exactly position new item
            adjustment = -remaining_scroll
            for vl in self.visual_lines:
                for cid in vl["lines"]:
                    self.canvas.move(cid, 0, adjustment)
                vl["y"] += adjustment
            self.scrolling = False
            self._draw_new_lines(wrapped_lines)
            if self.pending_text:
                text = self.pending_text
                self.pending_text = None
                self.add_text(text)
        else:
            self._cleanup_lines()
            self.overlay_wnd.after(frame_delay, lambda: self._scroll_old_lines(remaining_scroll,
                                                                              new_item_height,
                                                                              wrapped_lines))

    def _cleanup_lines(self):
        """Delete items that have scrolled out of the visible canvas."""
        new_visual_lines = []
        for vl in self.visual_lines:
            # bottom of the item
            if vl["y"] + vl["height"] > 0:
                new_visual_lines.append(vl)
            else:
                # remove all canvas IDs for this item
                for cid in vl["lines"]:
                    self.canvas.delete(cid)
        self.visual_lines = new_visual_lines
