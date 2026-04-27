import threading

from app.gui.widgets import captions_overlay


class CaptionsReceiver:
    def __init__(self, root_wnd, font_size, max_visible_lines, source_queue, gui_queue):
        self.source_queue = source_queue
        self.gui_queue = gui_queue
        self.captions_thread = None
        self.is_running = False
        self.last_partial = False

        self.overlay = captions_overlay.CaptionsOverlay(
            root=root_wnd,
            scroll_speed=400,
            max_visible_lines=max_visible_lines,
            font_size=font_size,
        )

    def start(self):
        if not self.is_running:
            self.captions_thread = threading.Thread(target=self.run)
            self.captions_thread.start()
            self.is_running = True

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            if complete:
                self.gui_queue.put(lambda t=text: self.send_complete(text=t))
            else:
                self.gui_queue.put(lambda t=text: self.send_partial(text=t))

    def send_complete(self, text):
        if self.overlay:
            if self.last_partial:
                self.overlay.update_last_text(text)
                self.last_partial = False
            else:
                self.overlay.add_text(text)

    def send_partial(self, text):
        if self.overlay:
            if self.last_partial:
                self.overlay.update_last_text(text)
            else:
                self.overlay.add_text(text)
                self.last_partial = True

    def stop(self):
        if self.is_running:
            self.source_queue.put((None, None))  # signal to stop
            self.captions_thread.join()
            self.captions_thread = None
            self.is_running = False
            self.overlay.destroy()
            self.overlay = None

__all__ = ["CaptionsReceiver"]

