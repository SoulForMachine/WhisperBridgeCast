import tkinter as tk
from typing import List


class GraphWidget(tk.Frame):
    """
    A lightweight graph widget that draws streaming float values on a Tkinter Canvas.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    max_points : int
        Maximum number of points to keep and display; older points are dropped.
    width : int
        Canvas width in pixels.
    height : int
        Canvas height in pixels.
    line_color : str
        Color of the graph polyline.
    border_color : str
        Color of the surrounding border rectangle.
    """

    def __init__(
        self,
        master,
        max_points: int = 200,
        width: int = 400,
        height: int = 200,
        line_color: str = "green",
        border_color: str = "black",
        bg_color: str = "white",
    ):
        super().__init__(master, width=width, height=height)

        self.max_points = max(1, int(max_points))
        self.width = int(width)
        self.height = int(height)
        self.line_color = line_color
        self.border_color = border_color

        self._values: List[float] = []
        # Fixed horizontal spacing; newest point stays at the right edge,
        # older points march leftward with the same pixel pitch (Task Manager style).
        self._x_step = (self.width - 1) / max(1, self.max_points - 1)

        # Create canvas that will hold both border and line.
        self.canvas = tk.Canvas(
            self,
            width=self.width,
            height=self.height,
            highlightthickness=0,
            bg=bg_color,
        )
        self.canvas.pack(fill="both", expand=True)

        # Border id is kept so we redraw it cheaply.
        self._border_id = None

        self._draw_border()

    def add_value(self, value: float):
        """Append a value and redraw the graph scaled to current min/max."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return

        self._values.append(v)
        if len(self._values) > self.max_points:
            # Drop oldest points to maintain a sliding window.
            self._values = self._values[-self.max_points :]
        self._redraw()

    def clear(self):
        """Remove all points and clear the graph area (border remains)."""
        self._values.clear()
        self._redraw(clear_only=True)

    def _draw_border(self):
        self.canvas.delete(self._border_id)
        self._border_id = self.canvas.create_rectangle(
            0,
            0,
            self.width - 1,
            self.height - 1,
            outline=self.border_color,
        )

    def _redraw(self, clear_only: bool = False):
        # Remove previous line drawings but keep the border.
        self.canvas.delete("graph-line")

        if clear_only or len(self._values) == 0:
            return

        n = len(self._values)
        points = []
        # Newest value on the right edge; older values scroll left with fixed spacing.
        right_x = self.width - 1
        for i, val in enumerate(reversed(self._values)):
            x = right_x - i * self._x_step
            y = self._scale_y(val)
            points.append((x, y))
        # Draw from oldest (leftmost) to newest (rightmost).
        points = list(reversed(points))

        if n == 1:
            r = 2
            x, y = points[0]
            self.canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                fill=self.line_color,
                outline=self.line_color,
                tags="graph-line",
            )
        else:
            coords = [c for pt in points for c in pt]
            self.canvas.create_line(
                *coords,
                fill=self.line_color,
                width=1,
                smooth=True,
                tags="graph-line",
            )

    def _scale_y(self, value: float) -> float:
        """Map value to canvas Y coordinate (inverted because Tk origin is top-left)."""
        vmin = min(self._values)
        vmax = max(self._values)

        if vmax == vmin:
            # Avoid division by zero; place the line in the vertical middle.
            return self.height / 2

        norm = (value - vmin) / (vmax - vmin)
        return (1.0 - norm) * (self.height - 1)


if __name__ == "__main__":
    # Minimal demo for manual testing.
    import random

    root = tk.Tk()
    root.title("GraphWidget demo")

    graph = GraphWidget(root, max_points=100, width=500, height=200, line_color="blue", border_color="gray")
    graph.pack(padx=10, pady=10)

    def tick():
        graph.add_value(random.uniform(-1, 1))
        root.after(100, tick)

    tick()
    root.mainloop()
