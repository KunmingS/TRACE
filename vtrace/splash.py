"""The start screen a bare `trace` prints.

A signpost, not a prompt: it names the handful of commands worth typing first and
ticks off the ones already done, so the next step is always the unticked one.
Every line is a real command, so whatever the reader copies goes through the
ordinary argument parser.

Rendered with Rich when it is installed — that is where the colour, width and
NO_COLOR handling come from — and as plain text when it is not.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# "ANSI Shadow"-style block letters, for terminals whose encoding carries them.
_BLOCK = """\
 ██╗   ██╗     ████████╗██████╗  █████╗  ██████╗███████╗
 ██║   ██║     ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██║   ██║ ███╗   ██║   ██████╔╝███████║██║     █████╗
 ╚██╗ ██╔╝ ╚══╝   ██║   ██╔══██╗██╔══██║██║     ██╔══╝
  ╚████╔╝         ██║   ██║  ██║██║  ██║╚██████╗███████╗
   ╚═══╝          ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝"""

_ASCII = r"""\
 __     __ _____ ____      _    ____ _____
 \ \   / /|_   _|  _ \    / \  / ___| ____|
  \ \ / / -| | | |_) |  / _ \| |   |  _|
   \ V /   | | |  _ <  / ___ \ |___| |___
    \_/    |_| |_| \_\/_/   \_\____|_____|"""

_TAGLINE = (
    "Video-based Temporal Recognition and Annotation of",
    "Continuous Ethograms of Animal Behavior",
)

# Kept in step with `cli.DEFAULT_GUI_PORT`; imported lazily so the start screen
# stays free of heavy imports.
def annotator_url() -> str:
    from vtrace.cli import DEFAULT_GUI_PORT

    return f"http://localhost:{DEFAULT_GUI_PORT}"

ACCENT = "cyan"


def trace_home() -> Path:
    root = os.environ.get("TRACE_HOME")
    return Path(root) if root else Path.home() / ".vtrace"


def demo_root() -> Path:
    # Delegated so the start screen ticks off the same directory the commands
    # actually use. `demo` is stdlib-only at import time; nothing here may pull
    # in torch, which would put seconds between typing `trace` and seeing it.
    from vtrace.demo import demo_dir

    return demo_dir()


def demo_steps():
    """(command, description, done) for each demo step.

    `done` is a STATE, not a history: it says the demo data is on this machine, so
    `demo download` has nothing left to do. `predict` and `train` are actions a user
    runs as often as they like — ticking them would claim they are finished, and
    then quietly discourage the second run. They carry `done=None`, which the
    renderers draw without a mark.

    The check is a filesystem stat — nothing here may import torch, which would put
    seconds between typing `trace` and seeing anything.
    """
    from vtrace.demo import missing_videos

    root = demo_root()
    try:
        have_data = not missing_videos(root, "test") and not missing_videos(root, "train")
    except OSError:
        have_data = False
    return (
        ("vtrace demo download", "the official CalMS21 videos + the detector", have_data),
        ("vtrace demo predict", "label a held-out video, write predictions", None),
        ("vtrace demo train", "train on the 70 official training videos", None),
    )


def _typed(command: str, interactive: bool) -> str:
    """How the reader should type `command` where they are standing.

    In the session the prompt already is `trace`, so echoing it back would have
    them type it twice; from a shell it is the executable's name and has to stay.
    """
    if not interactive:
        return command
    if command == "vtrace --help":
        return "help"
    return command.removeprefix("vtrace ")


def _unicode(encoding) -> bool:
    return "utf" in (encoding or "").lower()


def _art(encoding) -> str:
    return _BLOCK if _unicode(encoding) else _ASCII


def get_console():
    """A Rich console for stdout, or None when Rich is not installed."""
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console()


def _rich_screen(console, version: str, interactive: bool, annotator=None) -> None:
    from rich.table import Table
    from rich.text import Text

    tick = "✓" if _unicode(console.encoding) else "*"

    # One shared label width, so the three command blocks line up as a single
    # column instead of each grid measuring only its own rows.
    labels = [_typed(command, interactive) for command, _note, _done in demo_steps()]
    if annotator:
        # The URL shares this column; leaving it out truncates it to an ellipsis.
        labels.append(annotator)
    label_width = max(len(label) for label in labels) + 5  # tick + indent

    def rows(entries):
        table = Table.grid(padding=(0, 3))
        table.add_column(no_wrap=True, width=label_width)
        table.add_column(style="dim")
        for label, note in entries:
            table.add_row(label, note)
        return table

    console.print(Text(_art(console.encoding), style=ACCENT))
    for index, line in enumerate(_TAGLINE):
        suffix = f"   v{version}" if version and index == len(_TAGLINE) - 1 else ""
        console.print(Text(f" {line}{suffix}", style="dim"))

    console.print()
    if annotator:
        console.print(Text(" Annotator running — open it in Chrome or Edge", style="bold"))
        console.print(rows([(Text(f"   {annotator}", style="bold " + ACCENT),
                             "read and label videos on this computer")]))
    else:
        console.print(Text(" Annotate videos in your browser", style="bold"))
        console.print(rows([(Text(f"   {_typed('vtrace app', interactive)}", style="bold"),
                             "start the annotator")]))

    console.print()
    console.print(Text(" Try the demo", style="bold"))
    entries = []
    for command, note, done in demo_steps():
        # Assemble rather than concatenate: `Text + Text` would carry the mark's
        # style over the command, painting a finished step green instead of grey.
        label = Text.assemble(
            (f" {tick} ", "green") if done else ("   ", ""),
            (_typed(command, interactive), "dim" if done else "bold"),
        )
        entries.append((label, note))
    console.print(rows(entries))

    console.print()
    console.print(rows([(Text(f"   {_typed('vtrace --help', interactive)}", style="bold"),
                         "every command and flag")]))
    if interactive:
        console.print()
        console.print(Text(" Type a command below. Tab completes, ctrl-r searches "
                           "history, `exit` leaves.", style="dim"))
    console.print()


def _plain_screen(version: str, interactive: bool, stream, annotator=None) -> str:
    encoding = getattr(stream, "encoding", "")
    entries = [(f"   {annotator}", "read and label videos on this computer")
               if annotator
               else (f"   {_typed('vtrace app', interactive)}", "start the annotator")]
    entries += [((" * " if done else "   ") + _typed(command, interactive), note)
                for command, note, done in demo_steps()]
    entries += [(f"   {_typed('vtrace --help', interactive)}", "every command and flag")]
    width = max(len(label) for label, _ in entries)

    lines = [_art(encoding)]
    for index, line in enumerate(_TAGLINE):
        suffix = f"   v{version}" if version and index == len(_TAGLINE) - 1 else ""
        lines.append(f" {line}{suffix}")
    heading = (" Annotator running — open it in Chrome or Edge" if annotator
               else " Annotate videos in your browser")
    lines += ["", heading, f"{entries[0][0]:<{width}}   {entries[0][1]}"]
    lines += ["", " Try the demo"]
    lines += [f"{label:<{width}}   {note}" for label, note in entries[1:4]]
    lines += ["", f"{entries[4][0]:<{width}}   {entries[4][1]}", ""]
    if interactive:
        lines += [" Type a command below. Tab completes, ctrl-r searches history, "
                  "`exit` leaves.", ""]
    return "\n".join(lines)


def print_start_screen(version: str = "", *, interactive: bool = False, console=None,
                       annotator: str | None = None) -> None:
    """Render the start screen to stdout.

    `annotator` is the URL of an already-running annotator, when the caller
    started one; the screen then points at it instead of naming the command that
    would start it.
    """
    console = console or get_console()
    if console is None:
        print(_plain_screen(version, interactive, sys.stdout, annotator))
        return
    _rich_screen(console, version, interactive, annotator)
