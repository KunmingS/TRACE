"""One console for the lines a run puts in front of a person.

Rich already decides whether colour is wanted — NO_COLOR, a redirected stdout,
a terminal that cannot do it — and the steps that run as subprocesses inherit
FORCE_COLOR from the parent that is teeing them to a real terminal. So callers
here say what they mean and never ask whether colour is available.

This is deliberately separate from `verbosity.note`: that is for chatter the
reader has to opt into, this is for the handful of lines they always see.
"""
from __future__ import annotations

import re
import sys

# The one accent colour, shared with the start screen and the demo, so a run
# does not change palette as it moves from download to prep to training.
ACCENT = "cyan"

_MARKUP = re.compile(r"\[/?[a-z0-9 _.,#=-]*\]")

_console = None
_resolved = False


def console():
    """The shared Rich console, or None when Rich is not installed."""
    global _console, _resolved
    if not _resolved:
        _resolved = True
        try:
            from rich.console import Console
        except ImportError:
            _console = None
        else:
            # soft_wrap: these lines are paths and tables of numbers, and a
            # console that guesses 80 columns down a pipe would fold them.
            _console = Console(soft_wrap=True)
    return _console


def say(markup: str = "", plain: str | None = None) -> None:
    """Print one line, coloured when the terminal takes colour.

    `plain` is the same line without markup, for terminals (or installs) that
    get the uncoloured version; left out, the markup tags are simply removed.
    """
    active = console()
    if active is None:
        print(plain if plain is not None else _MARKUP.sub("", markup), flush=True)
        return
    active.print(markup, highlight=False)


def duration(seconds) -> str:
    """A length of time with its units spelled out.

    `1:28` beside a run that could plausibly take either is unreadable — an
    hour and a half or a minute and a half, no way to tell. Two units, always
    labelled, and the smallest one drops off once it stops mattering.
    """
    seconds = int(max(0.0, float(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def plain(markup: str) -> str:
    """`markup` with its tags removed — for log files and non-Rich output."""
    return _MARKUP.sub("", markup)


def is_terminal() -> bool:
    """Whether output is going somewhere that redraws a line in place."""
    active = console()
    if active is not None:
        return bool(active.is_terminal)
    return sys.stdout.isatty()


# ── the progress bar ─────────────────────────────────────────────────────────
# Every bar a run draws — training epochs, validation passes, proxy encodes —
# is drawn by this, so they all look like the same tool talking. It lives here
# rather than beside the training plot because prep needs it too, and prep must
# not import torch to draw a bar.

BAR_WIDTH = 22
_FULL = "━"
_HEAD = "╸"
_EMPTY = "━"


def bar_markup(fraction, width=BAR_WIDTH):
    """The progress bar itself, so every bar in a run looks like the others."""
    filled = int(fraction * width)
    if filled >= width:
        return f"[{ACCENT}]{_FULL * width}[/]"
    head = _HEAD if filled else ""
    body = _FULL * filled
    rest = _EMPTY * (width - filled - len(head))
    return f"[{ACCENT}]{body}{head}[/][dim]{rest}[/]"


# ── a line that is replaced rather than added to ─────────────────────────────
# Work that takes a minute should say so while it happens, and should not leave
# a minute of scrollback behind when it is done. `status` redraws one line in
# place; `clear_status` takes it away before the result is printed.
_status_width = 0


def status(markup: str) -> None:
    """Redraw one transient line. Silent when nothing can redraw it."""
    global _status_width
    if not is_terminal():
        return
    width = len(_MARKUP.sub("", markup))
    pad = " " * max(0, _status_width - width)
    _status_width = max(_status_width, width)
    active = console()
    if active is None:
        sys.stdout.write(_MARKUP.sub("", markup) + pad + "\r")
        sys.stdout.flush()
        return
    active.print(markup + pad, end="\r", highlight=False, crop=False)


def clear_status() -> None:
    """Wipe whatever `status` last drew."""
    global _status_width
    if not _status_width:
        return
    active = console()
    stream = active.file if active is not None else sys.stdout
    stream.write(" " * _status_width + "\r")
    stream.flush()
    _status_width = 0
