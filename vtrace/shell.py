"""The interactive session a bare `trace` opens at a terminal.

A read-eval-print loop over the ordinary argument parser: whatever is typed is
split into argv and handed to the same `main()` a one-shot `vtrace ...` call goes
through, so there is no second command surface to keep in sync.

The line editor is prompt_toolkit when it is installed — the same choice pgcli,
mycli and IPython make. Without it the loop falls back to `input()` plus
readline, which still gives history and arrow keys on any normal Unix box.

Neither completion nor reverse history search is offered. The command set is
short enough to read off the start screen, and a session whose whole surface is
printed on arrival does not need a second, hidden way to discover it. Up and
down still walk the history, which is the part people reach for.

The session opens on a bordered box, not the one-line prompt. The command most
people bring is the one the annotator's configuration page wrote — several lines
of `train --pairs … --output …` with `then eval …` after it — and a box is where
a pasted block of lines belongs. Enter runs it. Escape closes the box and leaves
the plain prompt, where `run` opens it again: two places to type, one command
surface. Pasting into the prompt still works too, on one line or with the
backslash continuations a copied shell command carries.

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

# The command set, and what the in-session `help` prints.
COMMANDS = {
    "app": "Serve the annotator UI (already running in this session)",
    "demo": "Download and run the CalMS21 walkthrough",
    "run": "Open the command box (paste the command the annotator wrote)",
    "train": "Train a model on video/CSV pairs",
    "eval": "Evaluate a trained model",
    "predict": "Run prediction on videos",
    "then": "Chain steps: train ... then eval ... then predict ...",
    "prepare": "Download model weights and local assets",
    "update": "Check PyPI for a newer release",
    "help": "Where the documentation lives",
    "exit": "Leave the session",
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
    "`run` opens the command box; esc closes it for this prompt.",
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

   The box the session opens on is for exactly this, drawn below the start
   screen so it stays in view: paste the chain the annotator wrote, or type
   one a step per line (ctrl-j for a new line), then enter runs it. Esc
   closes the box for the plain prompt, and `run` opens it again.
""")


# ── prompt_toolkit front end ─────────────────────────────────────────────────

def _make_session():
    """A prompt_toolkit session, or None when prompt_toolkit is missing."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style
    except ImportError:
        return None

    path = history_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(path))
    except OSError:
        history = None

    # Reverse-i-search is an emacs default binding, so removing the feature
    # means taking the key back rather than leaving an option unset.
    bindings = KeyBindings()

    @bindings.add("c-r")
    def _(event):
        pass

    style = Style.from_dict({"prompt": "bold cyan"})
    return PromptSession(
        history=history,
        style=style,
        key_bindings=bindings,
        enable_history_search=False,
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

    # Anything printed from here on is being read at the prompt, so a command
    # suggested to the reader drops the program's own name.
    splash.IN_SESSION = True
    try:
        return _loop(dispatch, session, console)
    finally:
        splash.IN_SESSION = False


def _loop(dispatch, session, console=None) -> int:
    """Read and run commands until the reader leaves.

    Two places to type. The session starts in the box, where a command copied
    from the annotator is pasted and enter runs it; escape closes the box and
    the loop falls back to the one-line prompt, where `run` opens the box again.
    Whichever place a command came from is where the loop returns after it.
    """
    in_box = True
    while True:
        if in_box:
            line = composer.compose()
            if line is None:
                in_box = False
                print("  Plain prompt. Type a command; `run` reopens the box; `exit` leaves.")
                continue
        else:
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
            in_box = True
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
    """Arrow keys and history for the fallback `input()` path.

    History only — no completion, and no reverse search.
    """
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
    # Tab inserts a tab, and ctrl-r does nothing: see the note at the top of
    # the module. Both are readline defaults, so both have to be taken back.
    readline.parse_and_bind("tab: self-insert")
    readline.parse_and_bind(r'"\C-r": ')
