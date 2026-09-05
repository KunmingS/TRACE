"""One switch for the build-time chatter the model emits.

Constructing a backbone prints a handful of facts that matter while developing
a model and matter to nobody running `vtrace predict`: parameter counts, which
checkpoint the weights came from, how many keys matched, whether the backbone
is frozen. They are worth keeping — the first question after a surprising
result is usually "which weights did it actually load" — but not worth putting
in front of someone labelling mice.

So they go through `note()`, which is silent unless asked. `TRACE_VERBOSE=1`
asks; so does `--verbose` on any command that offers it.
"""
from __future__ import annotations

import os
import sys

# Set from the environment once, then overridable in-process by `set_verbose`.
_VERBOSE = os.environ.get("TRACE_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def verbose() -> bool:
    """Whether build-time notes are being printed."""
    return _VERBOSE


def set_verbose(value: bool) -> None:
    """Turn build-time notes on or off for the rest of this process."""
    global _VERBOSE
    _VERBOSE = bool(value)


def note(message: str) -> None:
    """Print `message` only when the reader asked for this level of detail."""
    if _VERBOSE:
        print(message, file=sys.stderr, flush=True)


def warn(message: str) -> None:
    """Print `message` always: something loaded in a way worth questioning."""
    print(message, file=sys.stderr, flush=True)
