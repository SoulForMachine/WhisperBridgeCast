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
        self.overlay_wnd.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close button

        self.overlay_wnd.bind("<Button-1>", self._start_move)
        self.overlay_wnd.bind("<B1-Motion>", self._do_move)
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
        self.pending_text = []  # text waiting to be added
        self.scrolling = False

        # Set fixed window size
        self._set_window_geometry()

    # --- Window movement ---
    def _start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def _do_move(self, event):
        x = self.overlay_wnd.winfo_x() + (event.x - self.start_x)
        y = self.overlay_wnd.winfo_y() + (event.y - self.start_y)
        self.overlay_wnd.geometry(f"+{x}+{y}")

    # --- Window sizing and position ---
    def _set_window_geometry(self):
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
    def _wrap_text(self, text):
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
        self.pending_text.append((text, "add"))
        self._process_new_text()

    def update_last_text(self, text):
        self.pending_text.append((text, "update"))
        self._process_new_text()

    def destroy(self):
        self.overlay_wnd.destroy()

    # --- Internal ---
    def _process_new_text(self):
        if self.scrolling:
            return

        while self.pending_text:
            text, op = self.pending_text.pop(0)
            wrapped_lines = self._wrap_text(text)
            new_item_height = len(wrapped_lines) * self.line_height

            if op == "update" and self.visual_lines:
                # Remove last item's canvas lines
                for cid in self.visual_lines[-1]["lines"]:
                    self.canvas.delete(cid)
                # Remove from internal list
                self.visual_lines = self.visual_lines[:-1]

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
                break
            else:
                # Enough space, add immediately
                self._draw_new_lines(wrapped_lines)

    def _draw_new_lines(self, wrapped_lines):
        y = self.height - self.padding - len(wrapped_lines) * self.line_height
        line_ids = []
        for line in wrapped_lines:
            cid = self.canvas.create_text(self.padding, y, anchor="nw", text=line, fill="white", font=self.font)
            line_ids.append(cid)
            y += self.line_height
        self.visual_lines.append({
            "lines": line_ids,
            "y": self.height - self.padding - len(wrapped_lines) * self.line_height,
            "height": len(wrapped_lines) * self.line_height
        })
        self._cleanup_lines()

    def _scroll_old_lines(self, remaining_scroll, new_item_height, wrapped_lines):
        if remaining_scroll <= 0:
            self.scrolling = False
            self._draw_new_lines(wrapped_lines)
            self._process_new_text()
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
            self._process_new_text()
        else:
            self._cleanup_lines()
            self.overlay_wnd.after(frame_delay, lambda: self._scroll_old_lines(remaining_scroll,
                                                                              new_item_height,
                                                                              wrapped_lines))

    def _cleanup_lines(self):
        """Delete items that have scrolled out of the visible canvas."""
        """For visible items, hide wrapped lines that are above the padding."""
        new_visual_lines = []
        for vl in self.visual_lines:
            # The visual line (can be multiple wrapped lines) should be kept if its bottom is below the padding
            if vl["y"] + vl["height"] > self.padding:
                new_visual_lines.append(vl)
                y = vl["y"]
                for cid in vl["lines"]:
                    if y + self.line_height > self.padding:
                        self.canvas.itemconfig(cid, state="normal")
                    else:
                        self.canvas.itemconfig(cid, state="hidden")
                    y += self.line_height
            else:
                # remove all canvas IDs for this visual line
                for cid in vl["lines"]:
                    self.canvas.delete(cid)
        self.visual_lines = new_visual_lines
