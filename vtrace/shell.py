"""The interactive session a bare `trace` opens at a terminal.

A read-eval-print loop over the ordinary argument parser: whatever is typed is
split into argv and handed to the same `main()` a one-shot `vtrace ...` call goes
through, so there is no second command surface to keep in sync.

The line editor is prompt_toolkit when it is installed — the same choice pgcli,
mycli and IPython make — which brings a completion menu, reverse history search
and a status bar for free. Without it the loop falls back to `input()` plus
readline, which still gives history and arrow keys on any normal Unix box.

The prompt itself is one line. A chain is longer than that, and the answer is
`run`, which opens a bordered box to compose one in — so the prompt does not
also need a hidden multi-line mode, nor a permanent toolbar advertising one.
Pasting a chain still works, on one line or with the backslash continuations a
copied shell command carries.

The loop's job is to survive. Commands fail by calling `sys.exit()` (argparse
does it for a bad flag, the step runners for a non-zero exit code) and that has
to land back at the prompt instead of ending the session.
"""
from __future__ import annotations

import shlex
import signal
import sys
import traceback
from pathlib import Path

from vtrace import composer, splash

# The documentation, kept in one place. Also `Documentation` in pyproject.toml.
DOCS_URL = "https://kunmings.github.io/TRACE/"

_HISTORY_LIMIT = 1000
BACKSLASH = chr(92)
# `\` at the end of a line, with the newline: a pasted shell continuation.
_CONTINUATION = __import__("re").compile(BACKSLASH + BACKSLASH + r"\s*\n")

# Command tree, used for completion and for the in-session `help`.
COMMANDS = {
    "app": "Serve the annotator UI (already running in this session)",
    "demo": "Download and run the CalMS21 walkthrough",
    "run": "Open a box to compose a run of chained steps",
    "train": "Train a model on video/CSV pairs",
    "eval": "Evaluate a trained model",
    "predict": "Run prediction on videos",
    "then": "Chain steps: train ... then eval ... then predict ...",
    "prepare": "Download model weights and local assets",
    "update": "Check PyPI for a newer release",
    "help": "Where the documentation lives",
    "exit": "Leave the session",
}

_COMPLETION_TREE = {
    "app": {"--port", "--no-browser", "--verbose"},
    # No flags of its own: `run` opens the box, and the flags are typed in there.
    "run": None,
    "demo": {"download": {"--from"}, "predict": None, "train": None},
    "train": {"--model", "--config", "--pairs", "--output", "--eval-pairs",
              "--epochs", "--val-start-epoch", "--val-interval",
              "--input-resolution", "--resource-profile", "--pretrained",
              "--nproc", "--seed", "--resume", "--cfg-options"},
    "eval": {"--model-dir", "--pairs", "--resource-profile",
             "--nproc", "--seed", "--profile", "--auto-tune", "--cfg-options"},
    "predict": {"--model-dir", "--input", "--output", "--include-stems",
                "--threshold", "--resource-profile", "--profile", "--auto-tune"},
    "prepare": {"--weights"},
    "update": {"-y", "--yes", "--check-only", "--timeout"},
    "then": None,
    "help": {c: None for c in COMMANDS},
    "exit": None,
}


def history_file() -> Path:
    return splash.trace_home() / "history"


def _to_argv(line: str):
    """Split typed text into argv, or None when there is nothing to run.

    Accepts the commands exactly as the start screen prints them, the executable's
    own name and all, since that is what people paste. Newlines are plain
    whitespace to shlex, but a backslash before one is an escape that would
    survive as a token, so the continuations of a pasted command come off first.
    """
    line = _CONTINUATION.sub(" ", line)
    try:
        argv = shlex.split(line)
    except ValueError as exc:  # unbalanced quote
        print(f"  {exc}", file=sys.stderr)
        return None
    if argv and argv[0] in ("vtrace", "trace"):
        argv = argv[1:]
    return argv or None


# Only what the website cannot say, because it is about being in here.
_HELP_NOTES = (
    "Every command, flag and file format.",
    "",
    "`run` opens a box to compose a run.",
    "`COMMAND --help` prints one command's flags.",
    "Tab lists what can be typed here.",
)
_HELP_INDENT = " " * 19


def _print_help(console) -> None:
    """Point at the documentation rather than reprinting it.

    This used to be a table of every verb, which is a second copy of the website
    and of argparse's own help — two places to forget when a flag changes. The
    site is the one that can hold the whole story, so the session's job is to
    name it and to name the three things that only exist in here.
    """
    if console is None:
        print()
        print(f"   Documentation   {DOCS_URL}")
        for note in _HELP_NOTES:
            print(f"{_HELP_INDENT}{note}" if note else "")
        print()
        return

    from rich.text import Text

    console.print()
    console.print(Text.assemble(("   Documentation   ", "bold"),
                                (DOCS_URL, "bold " + splash.ACCENT)))
    for note in _HELP_NOTES:
        # no_wrap: these are aligned under the URL, and a wrap would drop the
        # continuation back to column zero and break the column.
        console.print(Text(f"{_HELP_INDENT}{note}" if note else "", style="dim"),
                      no_wrap=True, crop=True)
    console.print()


def _print_chain_help() -> None:
    """`help then`. There is no parser behind a separator, so this is written out."""
    print("""
   then — run steps one after another, stopping at the first failure.

     train --pairs VIDEO=CSV ... --output DIR then eval --pairs VIDEO=CSV
        One run. The eval videos score the model each epoch, and that is
        what writes best.pth. Without them, training keeps every epoch's
        checkpoint and calls none of them best.

     ... then predict --input DIR
        Reuses the model the training step just produced, so its timestamped
        run folder never has to be typed. Pass --model-dir to override.

     eval --model-dir DIR --pairs V=C then predict --input DIR
        An eval that does not follow a train keeps its ordinary meaning:
        score a model that already exists.

   `run` opens a bordered box for exactly this, drawn below the prompt so
   the screen above it stays: a step per line, ctrl-s to submit, esc or the
   Cancel button to leave. A chain can be laid out and read back before any
   of it starts.
""")


# ── prompt_toolkit front end ─────────────────────────────────────────────────

def _make_session():
    """A prompt_toolkit session, or None when prompt_toolkit is missing."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import NestedCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style
    except ImportError:
        return None

    path = history_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(path))
    except OSError:
        history = None

    style = Style.from_dict({"prompt": "bold cyan"})
    return PromptSession(
        completer=NestedCompleter.from_nested_dict(_COMPLETION_TREE),
        history=history,
        style=style,
        complete_while_typing=False,   # a menu that pops on every keystroke is noise
        enable_history_search=True,
    )


def _prompt_fragments():
    return [("class:prompt", "vtrace"), ("", " > ")]


# ── The loop ─────────────────────────────────────────────────────────────────

def run(dispatch, version: str = "") -> int:
    """Show the start screen, then read and run commands until the user leaves.

    `dispatch` is `vtrace.cli.main`; it is passed in rather than imported so
    the import stays one-directional.
    """
    # Start the annotator first, so the screen can name the address it is actually
    # listening on instead of an address the reader would have to go and create.
    from vtrace.cli import serve_in_background

    annotator = serve_in_background()

    console = splash.get_console()
    splash.print_start_screen(version, interactive=True, console=console,
                              annotator=annotator)

    session = _make_session()
    if session is None:
        _enable_readline()

    while True:
        try:
            if session is not None:
                line = session.prompt(_prompt_fragments())
            else:
                line = input("vtrace > ")
        except EOFError:  # Ctrl-D
            print()
            break
        except KeyboardInterrupt:  # Ctrl-C abandons the line, not the session
            continue

        argv = _to_argv(line)
        if argv is None:
            continue
        if argv[0] in ("exit", "quit", "q"):
            break
        # Interactive-only, so it never reaches the argument parser: `run` is a
        # way of typing a command, not a command of its own.
        if argv[0] == "run" and len(argv) == 1:
            composed = composer.compose()
            if composed is None:
                continue
            argv = _to_argv(composed)
            if argv is None:
                continue
        elif argv[0] == "run":
            # `run train ...` reads as "run this", and refusing it outright would
            # be pedantry — the rest of the line is already the command.
            argv = argv[1:]
        if argv[0] in ("help", "?") and len(argv) == 1:
            _print_help(console)
            continue
        if argv[0] in ("help", "?"):
            # `then` is a separator, so it has no parser to ask for --help.
            if argv[1:2] == ["then"]:
                _print_chain_help()
                continue
            argv = argv[1:] + ["--help"]

        # `vtrace app` installs its own SIGINT handler and never puts the old one
        # back, so without this the first Ctrl-C after running it would take the
        # whole session down instead of just clearing the prompt.
        previous_sigint = signal.getsignal(signal.SIGINT)
        try:
            dispatch(argv)
        except SystemExit as exc:
            # argparse exits on a bad flag or after --help; the step runners exit
            # with a failed subprocess's code; some handlers raise SystemExit with
            # a message instead of a code. All ordinary outcomes here — but a
            # message would be lost if it were not printed on the way past.
            if isinstance(exc.code, str):
                print(f"  {exc.code}", file=sys.stderr)
            elif isinstance(exc.code, int) and exc.code not in (0, 2):
                print(f"  (exit code {exc.code})", file=sys.stderr)
        except KeyboardInterrupt:
            # Ctrl-C during a command stops that command, not the session.
            print("\n  interrupted", file=sys.stderr)
        except Exception:
            traceback.print_exc()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)

    return 0


def _enable_readline():
    """Arrow keys and history for the fallback `input()` path."""
    try:
        import atexit
        import readline
    except ImportError:  # Windows without pyreadline
        return
    path = history_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            readline.read_history_file(path)
        readline.set_history_length(_HISTORY_LIMIT)
        atexit.register(readline.write_history_file, str(path))
    except OSError:
        pass
    readline.parse_and_bind("tab: complete")
    readline.set_completer(_readline_completer)
    readline.set_completer_delims(" \t\n")


def _readline_completer(text, state):
    matches = [name for name in _COMPLETION_TREE if name.startswith(text)]
    return matches[state] if state < len(matches) else None
