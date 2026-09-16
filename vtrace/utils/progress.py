"""The training loop's terminal display: a live loss plot over a progress bar.

A training log has always been a wall — one line every N steps, each with
losses, two learning rates and peak memory. That belongs in the run's log file.
What someone watching wants is the curve: is the loss still coming down, and
when does this finish. So the terminal gets a block that redraws itself in
place, and the file gets one line per epoch.

The block is drawn with braille (each character is a 2x4 grid of dots), which
gives eight times the resolution of block characters, and points are joined so
the result reads as a line rather than a bar chart. The x axis auto-scales to
the steps taken so far: the curve always spans the full width and every step
changes its shape, instead of a new column appearing once every few minutes.

    loss
     3.57 |⣄
          | ⠈⠢⣀
     2.09 |     ⠉⠒⠤⢄⣀
          |            ⠉⠑⠒⠤⠤⣀⣀
     0.61 |                    ⠉⠉⠉⠒⠒⠢⠤⠤⣀⣀
          +----------------------------------------
    Epoch 1/10  ---------  132/451  loss 3.109  eta 50:02

Redrawing means cursor-up escapes, and everything a step prints is teed to the
run's log file — so the block would otherwise leave every frame of its
animation in that file. It does not: the block is bracketed so the tee in
`steps._run` can drop exactly the lines that were redrawn (see `_LogTee`), and
what lands in the file is the one summary line each epoch ends with.

When output is not going to a terminal there is nothing to redraw into, so the
block is never drawn and only the summary lines are printed.
"""
from __future__ import annotations

import sys
import time

from vtrace.console import (
    ACCENT, BAR_WIDTH, bar_markup, clear_status, console,
    duration as _clock, is_terminal, plain, status,
)

__all__ = ["BAR_WIDTH", "bar_markup", "StatusBar", "EpochBar", "LossPlot",
           "BrailleCanvas", "plot_width", "PLOT_ROWS"]

# Plot geometry. The y-axis labels take a fixed gutter; the plot takes what is
# left, up to a width where more columns stop adding information.
PLOT_ROWS = 8
GUTTER = 7
_PLOT_MAX = 88
_PLOT_MIN = 24

# Braille: bit per dot, indexed [dy][dx] within a 2-wide, 4-tall cell.
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


class StatusBar:
    """A bar for a pass that should leave nothing behind when it finishes.

    Validation runs inside an epoch, under the epoch's own line, and what it
    has to say — 118 of 342 windows, another 46 seconds — stops being true the
    moment it ends. So it is drawn with `status`, which redraws one line in
    place and wipes it on `close`; the score that outlives it is printed by the
    caller.
    """

    def __init__(self, label, total, interval=0.25):
        self.label = label
        self.total = max(1, int(total))
        self.interval = interval
        self.started = time.monotonic()
        self.last_draw = 0.0

    def update(self, step):
        now = time.monotonic()
        if now - self.last_draw < self.interval and step < self.total:
            return
        self.last_draw = now
        elapsed = now - self.started
        eta = ""
        if step:
            remaining = elapsed / step * (self.total - step)
            eta = f"  [dim]eta {_clock(remaining)}[/]"
        width = len(str(self.total))
        status(f"  [dim]{self.label}[/]  {bar_markup(step / self.total)}  "
               f"[dim]{step:>{width}d}/{self.total}[/]{eta}")

    def close(self):
        clear_status()


def plot_width():
    """How wide the plot fits, or 0 when the terminal is too narrow for one."""
    active = console()
    total = active.width if active is not None else 80
    width = min(_PLOT_MAX, total - GUTTER - 3)
    return width if width >= _PLOT_MIN else 0


class BrailleCanvas:
    """A dot canvas `cells` wide and `rows` tall, at 2x4 dots per cell."""

    def __init__(self, cells, rows):
        self.cells = cells
        self.rows = rows
        self.dx = cells * 2
        self.dy = rows * 4
        self.grid = [[0] * cells for _ in range(rows)]

    def set(self, x, y):
        if not (0 <= x < self.dx and 0 <= y < self.dy):
            return
        self.grid[y // 4][x // 2] |= _DOTS[y % 4][x % 2]

    def line(self, x0, y0, x1, y1):
        """Join two points, so a sparse series still reads as a curve."""
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            self.set(x0, y0)
            return
        for i in range(steps + 1):
            t = i / steps
            self.set(round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t))

    def rows_out(self):
        return ["".join(chr(0x2800 + bits) for bits in row) for row in self.grid]


class LossPlot:
    """Every step's loss, drawn as a curve over the run so far.

    Keeping the raw per-step values (a few thousand floats) rather than
    pre-binned buckets is what lets the x axis rescale as the run grows: each
    redraw re-bins into however many dot columns the terminal has.
    """

    def __init__(self, cells, rows=PLOT_ROWS):
        self.cells = cells
        self.rows = rows
        self.values = []

    @property
    def active(self):
        return self.cells >= _PLOT_MIN

    def add(self, loss):
        if loss is not None:
            self.values.append(float(loss))

    def _columns(self):
        """The series re-binned to one mean per dot column, left to right."""
        count = len(self.values)
        width = self.cells * 2
        if count <= width:
            return list(self.values)
        sums = [0.0] * width
        counts = [0] * width
        for index, value in enumerate(self.values):
            column = min(width - 1, index * width // count)
            sums[column] += value
            counts[column] += 1
        return [sums[i] / counts[i] for i in range(width) if counts[i]]

    def render(self):
        """The plot as a list of Rich-markup lines (y axis included)."""
        if not self.active or not self.values:
            return []
        series = self._columns()
        canvas = BrailleCanvas(self.cells, self.rows)
        low, high = min(series), max(series)
        span = high - low
        span_dx = canvas.dx - 1
        span_dy = canvas.dy - 1

        def point(index):
            x = 0 if len(series) == 1 else round(index / (len(series) - 1) * span_dx)
            if span <= 0:
                return x, span_dy // 2
            return x, round((high - series[index]) / span * span_dy)

        previous = point(0)
        canvas.set(*previous)
        for index in range(1, len(series)):
            current = point(index)
            canvas.line(*previous, *current)
            previous = current

        # Labels on the top, middle and bottom rows; the rest keep the axis.
        labels = {0: high, self.rows // 2: (high + low) / 2, self.rows - 1: low}
        lines = []
        for index, row in enumerate(canvas.rows_out()):
            if index in labels:
                gutter = f"[dim]{labels[index]:>{GUTTER}.3f}[/][dim]┤[/]"
            else:
                gutter = f"{'':>{GUTTER}}[dim]│[/]"
            lines.append(f"{gutter}[{ACCENT}]{row}[/]")
        lines.append(f"{'':>{GUTTER}}[dim]└{'─' * self.cells}[/]")
        return lines


class EpochBar:
    """The live block for one epoch: the loss plot above, the bar below.

    `update(step, loss)` is safe to call every iteration — redraws are throttled
    to `interval` seconds, so the cost of the rest is a clock read.
    """

    def __init__(self, epoch, total_epochs, steps, plot=None, interval=0.25):
        self.epoch = epoch
        # Padded to the widest label the run will produce, so "Epoch 10/10" does
        # not shove the bar a column right on the last epoch.
        width = len(f"Epoch {total_epochs}/{total_epochs}")
        self.label = f"Epoch {epoch + 1}/{total_epochs}".ljust(width)
        self.steps = max(1, int(steps))
        self.plot = plot
        self.interval = interval
        self.started = time.monotonic()
        self.last_draw = 0.0
        self.last_step_drawn = -1
        self.step = 0
        self.loss = None
        self.live = is_terminal()
        self.rows_drawn = 0

    # ── the bar line ──

    def _line(self, done):
        fraction = min(1.0, self.step / self.steps)
        elapsed = time.monotonic() - self.started
        width = len(str(self.steps))
        parts = [f"[bold]{self.label}[/]", bar_markup(fraction),
                 f"[dim]{self.step:>{width}d}/{self.steps}[/]"]
        if self.loss is not None:
            parts.append(f"loss [bold]{self.loss:.3f}[/]")
        if done:
            parts.append(f"[dim]{_clock(elapsed)}[/]")
        elif self.step:
            remaining = elapsed / self.step * (self.steps - self.step)
            parts.append(f"[dim]eta {_clock(remaining)}[/]")
        return "  ".join(parts)

    # ── drawing the block ──

    def _out(self):
        active = console()
        return active.file if active is not None else sys.stdout

    def _rewind(self):
        """Move back over the block just drawn and clear it.

        The cursor-up is what the log tee keys on to drop the redrawn lines, so
        it is emitted at the start of a line and nowhere else.
        """
        if not self.rows_drawn:
            return
        stream = self._out()
        stream.write(f"\x1b[{self.rows_drawn}A\x1b[J")
        stream.flush()
        self.rows_drawn = 0

    def _draw(self, lines):
        active = console()
        if active is None:
            stream = self._out()
            stream.write("\n".join(plain(line) for line in lines) + "\n")
            stream.flush()
        else:
            for line in lines:
                active.print(line, highlight=False, crop=False)
        self.rows_drawn = len(lines)

    # ── the loop's two calls ──

    def update(self, step, loss=None, step_loss=None):
        self.step = step
        if loss is not None:
            self.loss = loss
        if self.plot is not None:
            self.plot.add(step_loss if step_loss is not None else loss)
        if not self.live:
            return
        now = time.monotonic()
        if now - self.last_draw < self.interval and step < self.steps:
            return
        # A step can take seconds, and redrawing ten lines four times a second
        # to advance nothing but the clock is noise on the pipe. Once the step
        # is already on screen, slow down to a tick.
        if step == self.last_step_drawn and now - self.last_draw < 1.0:
            return
        self.last_draw = now
        self.last_step_drawn = step
        lines = []
        if self.plot is not None:
            lines.extend(self.plot.render())
        lines.append(self._line(done=False))
        self._rewind()
        self._draw(lines)

    def close(self):
        """Replace the block with the one line the epoch is worth keeping."""
        self._rewind()
        summary = self._line(done=True)
        active = console()
        if active is None:
            stream = self._out()
            stream.write(plain(summary) + "\n")
            stream.flush()
        else:
            active.print(summary, highlight=False, crop=False)
