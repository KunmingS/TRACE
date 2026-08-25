"""The interactive session a bare `trace` opens at a terminal.

A read-eval-print loop over the ordinary argument parser: whatever is typed is
split into argv and handed to the same `main()` a one-shot `vtrace ...` call goes
through, so there is no second command surface to keep in sync.

The line editor is prompt_toolkit when it is installed — the same choice pgcli,
mycli and IPython make — which brings a completion menu, reverse history search
and a status bar for free. Without it the loop falls back to `input()` plus
readline, which still gives history and arrow keys on any normal Unix box.

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

from vtrace import splash

_HISTORY_LIMIT = 1000

# Command tree, used for completion and for the in-session `help`.
COMMANDS = {
    "app": "Serve the annotator UI (already running in this session)",
    "demo": "Download and run the CalMS21 walkthrough",
    "train": "Train a model on video/CSV pairs",
    "eval": "Evaluate a trained model",
    "predict": "Run prediction on videos",
    "pipeline": "prep -> train -> test -> predict in one go",
    "prepare": "Download model weights and local assets",
    "update": "Check PyPI for a newer release",
    "help": "This list (`help COMMAND` for flags)",
    "exit": "Leave the session",
}

_COMPLETION_TREE = {
    "app": {"--host", "--port", "--dev"},
    "demo": {"download": {"--from"}, "predict": None, "train": None},
    "train": {"--model", "--config", "--work-dir", "--pairs", "--nproc", "--cfg-options"},
    "eval": {"--model-dir", "--work-dir", "--pairs", "--cfg-options"},
    "predict": {"--model-dir", "--input", "--output", "--threshold"},
    "pipeline": {"--train", "--extra-test", "--infer", "--model", "--config",
                 "--work-dir", "--pairs", "--input", "--epochs", "--resource-profile"},
    "prepare": {"--weights"},
    "update": {"-y", "--yes", "--check-only", "--timeout"},
    "help": {c: None for c in COMMANDS},
    "exit": None,
}


def history_file() -> Path:
    return splash.trace_home() / "history"


def _to_argv(line: str):
    """Split a typed line into argv, or None when there is nothing to run.

    Accepts the commands exactly as the start screen prints them, the executable's
    own name and all, since that is what people paste.
    """
    try:
        argv = shlex.split(line)
    except ValueError as exc:  # unbalanced quote
        print(f"  {exc}", file=sys.stderr)
        return None
    if argv and argv[0] in ("vtrace", "trace"):
        argv = argv[1:]
    return argv or None


def _print_help(console) -> None:
    """The in-session command list.

    Deliberately not argparse's `--help`: that one is a wall of usage strings
    built for a shell prompt, while in here the reader already knows they are in
    `trace` and wants to see what verbs exist.
    """
    if console is None:
        width = max(len(name) for name in COMMANDS)
        print()
        for name, note in COMMANDS.items():
            print(f"   {name:<{width}}   {note}")
        print("\n   Add --help to any command for its flags.\n")
        return

    from rich.table import Table
    from rich.text import Text

    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(style="dim")
    for name, note in COMMANDS.items():
        table.add_row(f"   {name}", note)
    console.print()
    console.print(table)
    console.print(Text("   Add --help to any command for its flags.", style="dim"))
    console.print()


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

    style = Style.from_dict({
        "prompt": "bold cyan",
        "bottom-toolbar": "fg:#888888 bg:#202020",
    })
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
        if argv[0] in ("help", "?") and len(argv) == 1:
            _print_help(console)
            continue
        if argv[0] in ("help", "?"):
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
