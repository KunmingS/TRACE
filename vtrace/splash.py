"""The start screen a bare `trace` prints.

A menu, not a signpost. It names the handful of commands worth typing first and
ticks off the ones already done, so the next step is always the unticked one.
Every line is a real command, so whatever the reader copies goes through the
ordinary argument parser.

The commands are one list under one sentence that says they are there to be
picked. They used to sit under headings — "Train a model, or predict videos",
"Try the demo" — which read as prose about the tool rather than as a menu, and
left the reader to work out that the bold words were things to type. Groups are
still grouped, by a blank line and nothing else.

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

# The line that turns a list of commands into a menu. Two versions, because the
# session's prompt runs them and a shell's does not.
_PICK_IN_SESSION = " Type one of these and press enter:"
_PICK_IN_SHELL = " Run any of these:"


def _pick_line(interactive: bool) -> str:
    return _PICK_IN_SESSION if interactive else _PICK_IN_SHELL


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
        ("vtrace demo predict", "label the 19 held-out test videos, write predictions", None),
        ("vtrace demo train", "train on the 70 official training videos, score the test split", None),
    )


# `run` only exists at the prompt: it opens the command box the session starts
# in, which is a way of typing a command and has nothing to do outside a
# session. The screen is printed in both places, so the row appears only in the
# interactive one.
RUN_STEP = ("run", "open the command box: paste what the annotator wrote")

# Also prompt-only, and for the same reason: there is nothing to leave when the
# screen is printed from a shell. It earns a row rather than a mention in the
# footer — how to get out is the one thing a reader should never have to hunt
# for, and a dim line under everything else is where it goes unread.
EXIT_STEP = ("exit", "leave the session")


# `update` appears only when there is something to update to. A permanent row
# saying "check for a newer release" asks the reader to do the checking; the
# session has already done it by the time this screen is drawn, so the row is
# either news or absent. Typing `update` still works either way.
def update_step(latest: str, current: str):
    """The (command, note) row announcing `latest`, for a reader on `current`."""
    note = f"V-TRACE {latest} is out"
    if current:
        note += f" — you have {current}"
    return ("vtrace update", note)


# Set by `vtrace.shell.run` for as long as the prompt is what runs commands.
# Anything that prints a command for the reader to type next has to know which
# of the two places they are standing in: inside the session the program's own
# name is already the prompt, so echoing it back tells them to type it twice.
IN_SESSION = False


def typed(command: str) -> str:
    """`command` written the way the reader should type it, wherever they are."""
    return _typed(command, IN_SESSION)


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


def _rich_screen(console, version: str, interactive: bool, annotator=None,
                 update: str | None = None) -> None:
    from rich.table import Table
    from rich.text import Text

    tick = "✓" if _unicode(console.encoding) else "*"

    # One shared label width, so the three command blocks line up as a single
    # column instead of each grid measuring only its own rows.
    labels = [_typed(command, interactive) for command, _note, _done in demo_steps()]
    if update:
        labels.append(_typed(update_step(update, version)[0], interactive))
    if interactive:
        labels.append(f"   {RUN_STEP[0]}")
        labels.append(f"   {EXIT_STEP[0]}")
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
    console.print(Text(_pick_line(interactive), style="bold"))

    if update:
        command, note = update_step(update, version)
        console.print()
        console.print(rows([(Text.assemble(
            (f" {'↑' if _unicode(console.encoding) else '!'} ", "bold " + ACCENT),
            (_typed(command, interactive), "bold"),
        ), note)]))

    if interactive:
        console.print()
        console.print(rows([(Text(f"   {RUN_STEP[0]}", style="bold"), RUN_STEP[1])]))

    console.print()
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
    closing = [(Text(f"   {_typed('vtrace --help', interactive)}", style="bold"),
                "documentation, on the web")]
    if interactive:
        closing.append((Text(f"   {EXIT_STEP[0]}", style="bold"), EXIT_STEP[1]))
    console.print(rows(closing))
    if interactive:
        console.print()
        # no_wrap, like the help notes: a wrap would drop the tail of the
        # sentence to column zero, under the art rather than under the text.
        console.print(Text.assemble(
            (" The box below is ", "dim"), ("run", "bold " + ACCENT),
            (" — paste what the annotator wrote, or type one.", "dim")),
            no_wrap=True, crop=True)
        console.print(Text(" Esc leaves the box and comes back to this list.",
                           style="dim"), no_wrap=True, crop=True)
    console.print()


def _plain_screen(version: str, interactive: bool, stream, annotator=None,
                  update: str | None = None) -> str:
    encoding = getattr(stream, "encoding", "")
    entries = [(f"   {annotator}", "read and label videos on this computer")
               if annotator
               else (f"   {_typed('vtrace app', interactive)}", "start the annotator")]
    update_entries = []
    if update:
        command, note = update_step(update, version)
        update_entries = [(f" ! {_typed(command, interactive)}", note)]
    entries += update_entries
    run_entries = [(f"   {RUN_STEP[0]}", RUN_STEP[1])] if interactive else []
    entries += run_entries
    entries += [((" * " if done else "   ") + _typed(command, interactive), note)
                for command, note, done in demo_steps()]
    closing = [(f"   {_typed('vtrace --help', interactive)}",
                "documentation, on the web")]
    if interactive:
        closing.append((f"   {EXIT_STEP[0]}", EXIT_STEP[1]))
    entries += closing
    width = max(len(label) for label, _ in entries)

    lines = [_art(encoding)]
    for index, line in enumerate(_TAGLINE):
        suffix = f"   v{version}" if version and index == len(_TAGLINE) - 1 else ""
        lines.append(f" {line}{suffix}")
    heading = (" Annotator running — open it in Chrome or Edge" if annotator
               else " Annotate videos in your browser")
    lines += ["", heading, f"{entries[0][0]:<{width}}   {entries[0][1]}"]
    lines += ["", _pick_line(interactive)]
    if update_entries:
        lines += [""]
        lines += [f"{label:<{width}}   {note}" for label, note in update_entries]
    if run_entries:
        lines += [""]
        lines += [f"{label:<{width}}   {note}" for label, note in run_entries]
    demo_start = 1 + len(update_entries) + len(run_entries)
    lines += [""]
    lines += [f"{label:<{width}}   {note}"
              for label, note in entries[demo_start:demo_start + 3]]
    lines += [""]
    lines += [f"{label:<{width}}   {note}" for label, note in closing]
    lines += [""]
    if interactive:
        lines += [" The box below is `run` — paste what the annotator wrote, or type one.",
                  " Esc leaves the box and comes back to this list.", ""]
    return "\n".join(lines)


def print_start_screen(version: str = "", *, interactive: bool = False, console=None,
                       annotator: str | None = None, update: str | None = None) -> None:
    """Render the start screen to stdout.

    `annotator` is the URL of an already-running annotator, when the caller
    started one; the screen then points at it instead of naming the command that
    would start it. `update` is a newer version on PyPI, when one was found —
    see `vtrace.version_check`, which is what decides whether to look.
    """
    console = console or get_console()
    if console is None:
        print(_plain_screen(version, interactive, sys.stdout, annotator, update))
        return
    _rich_screen(console, version, interactive, annotator, update)
