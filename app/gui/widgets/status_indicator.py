import tkinter as tk
from tkinter import ttk


class StatusIndicator(ttk.Frame):
    def __init__(self, parent, states, size=12, **kwargs):
        super().__init__(parent, **kwargs)

        if not states:
            raise ValueError("states cannot be empty")

        self.size = size

        # parse states
        self.states = {}
        for s in states:
            if len(s) == 2:
                name, color = s
                blink = None
            elif len(s) == 3:
                name, color, blink = s
            else:
                raise ValueError("State must be (name, color) or (name, color, blink_delay)")
            self.states[name] = (color, blink)

        self.default_state = states[0][0]
        self.current_state = self.default_state

        self.canvas = tk.Canvas(
            self,
            width=size,
            height=size,
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack()

        pad = 1
        self.circle = self.canvas.create_oval(
            pad, pad, size - pad, size - pad,
            fill=self.states[self.default_state][0],
            outline=""
        )

        self._blink_job = None
        self._blink_visible = True

    def _set_color(self, color):
        self.canvas.itemconfigure(self.circle, fill=color)

    def _blink_step(self):
        color, delay = self.states[self.current_state]

        if self._blink_visible:
            self._set_color(color)
        else:
            default_color = self.states[self.default_state][0]
            self._set_color(default_color)

        self._blink_visible = not self._blink_visible
        self._blink_job = self.after(int(delay * 1000), self._blink_step)

    def set_state(self, name):
        if name not in self.states:
            raise ValueError(f"Unknown state: {name}")

        if self._blink_job:
            self.after_cancel(self._blink_job)
            self._blink_job = None

        self.current_state = name
        color, delay = self.states[name]

        if delay is None:
            self._set_color(color)
        else:
            self._blink_visible = True
            self._blink_step()

    def get_state(self):
        return self.current_state
