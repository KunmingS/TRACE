"""The framed box the interactive session opens for a command.

The session opens on it. The command most people bring to the prompt was
written by the annotator's configuration page and copied: three or four lines
of `train --pairs … --output …`, with `then eval …` after it. A one-line prompt
asks the reader to believe a text area is hiding there; this draws the box, so
there is nothing to believe — a bordered field with a title, a hint under it,
and a Run and a Close button that tab reaches. Ordinary commands
(`demo download`, `help`) run from the same box, so nobody has to know in advance
which kind of command they are about to type.

The box carries its own name. The start screen is a list of commands to pick
from and this is what `run` opens, so the frame's title says `run`: a bordered
field with no name is one the reader has to infer a purpose for, and the session
opens straight into it before anything has been picked.

Escape goes back — out of the box to the plain one-line prompt, where `run` opens
it again. Two places to type, one command surface: whatever comes out of either
is split into argv and handed to the same `main()` a one-shot `vtrace …` call
goes through. Nothing is executed here; this module only collects a command.

Enter runs. A pasted command arrives with its own newlines (terminals bracket a
paste, so the newlines inside it are text, not key presses), which is why the
key that finishes typing can be the key everyone reaches for. A new line typed
by hand is ctrl-j or alt-enter; the box's own example shows the shape.

Drawn inline, below the start screen, rather than on a screen of its own. The
session is a transcript — the start screen, the address the annotator is
listening on, whatever the last command printed — and a full-screen box would
hide all of it at the moment the reader is composing the next command against it.

The mouse is left to the terminal. prompt_toolkit can take it over (clickable
buttons, wheel-scrolling inside the field), but taking it over means the
terminal stops doing its own job with it: while the box is up, nothing on the
screen can be selected and the scrollback cannot be scrolled — and the box is up
whenever the session is waiting for input, which is exactly when someone wants
to scroll back over what the last command printed or copy a path out of it. So
mouse tracking stays off. The buttons are reached with tab, and their key is
printed on them.

Escape cancels, but the binding is deliberately NOT eager. An eager one fires on
the first escape byte, and every sequence a terminal sends begins with one — an
arrow key, a function key, a bracketed paste. Without eager, prompt_toolkit
waits to see whether more bytes arrive that form a known sequence, and only a
genuinely lone escape key reaches the handler.

Tab moves focus, so the two buttons are reachable from the keyboard.
"""
from __future__ import annotations

TITLE = " run  —  paste the command from the annotator, or type one "
# The fullest form worth copying: two training pairs, where the run goes, the
# evaluation videos that decide best.pth, and a prediction pass over new
# footage. No schedule and no resolution — the model preset already carries
# both, and naming them here would teach a number nobody chose.
#
# A line per keyword, then a line per group of flags, which is the shape the
# configuration page writes. Newlines are whitespace to the parser, so the
# shape is free: it costs nothing and it makes the three steps and the `then`
# between them countable at a glance.
EXAMPLE = """train
--model maev2b --output /data/runs
--pairs /data/a.mp4=/data/a.csv /data/b.mp4=/data/b.csv
then
eval
--pairs /eval/c.mp4=/eval/c.csv
then
predict
--input /data/new --threshold 0.25"""
HINT = "enter runs it  ·  ctrl-j starts a new line  ·  esc goes back to the list"
# Wide enough for a pair of absolute paths either side of an `=` without
# wrapping, and narrow enough to sit inside a modest terminal.
_WIDTH = 92
_TEXT_HEIGHT = 6
# See where this is applied, below.
_ESCAPE_WAIT = 0.15
# Matches the start screen's accent, so the box reads as part of the same tool.
ACCENT = "#33d6d6"


def available() -> bool:
    """Whether the framed box can be drawn at all."""
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return True


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.split("\n"))


def compose(initial: str = "") -> str | None:
    """Show the box and return what was typed, or None when it was closed.

    A blank box does not run and does not close: enter on nothing does nothing,
    so a stray key never drops the reader out to the prompt by accident. Closing
    is escape (or the button), and it is the only way None comes back.
    """
    if not available():
        return _compose_plain(initial)

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.dimension import D
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Button, Frame, Label, TextArea

    result: dict[str, str | None] = {"text": None}

    text_area = TextArea(
        text=initial,
        multiline=True,
        wrap_lines=True,
        scrollbar=True,
        height=D.exact(_TEXT_HEIGHT),
        style="class:composer.text",
    )

    def submit() -> None:
        if not text_area.text.strip():
            return
        result["text"] = text_area.text
        application.exit()

    def cancel() -> None:
        result["text"] = None
        application.exit()

    # Bound on the text control itself, not the application: the focused
    # control's bindings win, and the multiline buffer's own `enter` would
    # otherwise insert a newline before this ever ran.
    text_keys = KeyBindings()

    @text_keys.add("enter")
    def _(event):
        submit()

    @text_keys.add("c-j")
    @text_keys.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    text_area.control.key_bindings = text_keys

    bindings = KeyBindings()

    # Run and close from anywhere, including with the cursor in the text, so
    # the buttons are an option rather than a detour.
    @bindings.add("c-s")
    def _(event):
        submit()

    # Not eager: see the note at the top of the module. A lone escape closes; an
    # escape that turns out to be the start of a terminal sequence does not.
    @bindings.add("c-c")
    @bindings.add("escape")
    def _(event):
        cancel()

    @bindings.add("tab")
    def _(event):
        event.app.layout.focus_next()

    @bindings.add("s-tab")
    def _(event):
        event.app.layout.focus_previous()

    box = HSplit([
        Frame(
            HSplit([
                Label(text="  example", style="class:composer.example.label",
                      dont_extend_height=True),
                Label(text=_indent(EXAMPLE, "  "), style="class:composer.example",
                      dont_extend_height=True),
                # A rule, so the line where typing starts is unmistakable: the
                # example above it is the same shape as the thing being typed,
                # and without a divider the two read as one block.
                Window(height=1, char="─", style="class:composer.rule"),
                text_area,
                Window(height=1, char=" "),
                Label(text="  " + HINT, style="class:composer.hint",
                      dont_extend_height=True),
            ]),
            title=TITLE,
            style="class:composer.frame",
        ),
        # The key is printed on the button rather than in a legend beside it: a
        # button reached by tab still leaves the question of what does the same
        # job without leaving the text, and that answer belongs on the button
        # itself.
        VSplit([
            Window(width=2, char=" "),
            Button("Run  enter", handler=submit, width=16),
            Window(width=2, char=" "),
            Button("Back  esc", handler=cancel, width=16),
            Window(),
        ], height=1),
    ], width=D.exact(_WIDTH))

    style = Style.from_dict({
        "composer.frame frame.border": ACCENT,
        "composer.frame frame.label": "bold " + ACCENT,
        "composer.text": "",
        "composer.hint": "#7f7f7f",
        "composer.example.label": "#5f5f5f",
        "composer.example": "#6f8f8f",
        "composer.rule": "#3a3a3a",
        "button.text": "#c6c6c6",
        "button.arrow": ACCENT + " bold",
        "button.focused": "bg:" + ACCENT,
        "button.focused button.text": "bg:" + ACCENT + " #062028 bold",
        "button.focused button.arrow": "bg:" + ACCENT + " #062028 bold",
    })

    application = Application(
        # The filler window is what keeps the box `_WIDTH` wide: without it the
        # HSplit is the whole layout and stretches to the terminal's width.
        layout=Layout(VSplit([box, Window()]), focused_element=text_area),
        key_bindings=bindings,
        style=style,
        # Off, so the terminal keeps selection and scrollback: see the note at
        # the top of the module.
        mouse_support=False,
        # Inline: the box is printed where the cursor already is and the
        # transcript above it stays on screen.
        full_screen=False,
    )
    # How long a lone `escape` waits to see whether more bytes are coming that
    # would make it the start of a sequence. Instance attributes, not
    # constructor arguments, so they are set here. The defaults (0.5s and 1.0s)
    # are spent on every single close, and a key that does nothing for a second
    # reads as a dead key — while a terminal's own sequences arrive as one burst,
    # their bytes in the same read, so a fraction of that still separates them.
    application.ttimeoutlen = _ESCAPE_WAIT
    application.timeoutlen = _ESCAPE_WAIT
    application.run()

    typed = (result["text"] or "").strip()
    return typed or None


def _compose_plain(initial: str = "") -> str | None:
    """The fallback for a box that cannot be drawn.

    Without prompt_toolkit there is no way to frame anything, so it takes the
    command one line at a time and runs it on a blank line — the same shape,
    minus the border. A pasted command still lands whole: each of its lines is
    read in turn, and the blank line after it is what runs it.
    """
    print()
    print("  Paste the command from the annotator, or type one. End with a blank line.")
    print("  A blank first line closes the box and leaves the plain prompt.")
    print()
    lines = [initial] if initial else []
    while True:
        try:
            line = input("  | ")
        except EOFError:
            print()
            break
        if not line.strip():
            break
        lines.append(line)
    typed = "\n".join(lines).strip()
    return typed or None
