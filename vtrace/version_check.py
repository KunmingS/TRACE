"""Whether a newer V-TRACE is on PyPI, answered without making anyone wait.

The start screen is the one place a new release can be mentioned to somebody who
would otherwise never look, so the session checks on the way up. What it must not
do is cost the reader anything for the privilege: a network round trip on every
launch would put a second between typing `vtrace` and seeing it, and a laptop off
the network would pay a timeout instead.

Two things keep it free. The answer is cached in `~/.vtrace/update-check.json`
and only refreshed once a day, so almost every launch reads a file. And the
refresh runs on a daemon thread started *before* the annotator's server, so on
the launches that do check, the request overlaps work the session was doing
anyway and the answer is usually there by the time the screen is drawn. When it
is not, the screen simply says nothing and the answer lands in the cache for
next time — a version notice is not worth a spinner.

Nothing here raises. PyPI being unreachable, a proxy in the way, a read-only home
directory: all of them mean "no notice this time", never a traceback in front of
someone who asked to label some videos.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# How long a fetched answer is trusted before another fetch is worth making.
# A day: new releases are not hourly, and this runs on every single launch.
CACHE_TTL = 24 * 60 * 60

# What the background fetch is given before it is abandoned. Shorter than the
# `update` command's own timeout — that one was asked for and can wait, this one
# is speculative and must never be why a start screen is late.
FETCH_TIMEOUT = 4.0

# How long `pending` will wait for a fetch that is still in flight. Missing the
# deadline costs nothing but a day's delay on the notice, so it is set by what a
# person will accept before a start screen appears rather than by what a slow
# connection needs.
DEFAULT_WAIT = 1.0

_thread = None
_result = {}
_lock = threading.Lock()


def cache_path() -> Path:
    from vtrace.splash import trace_home

    return trace_home() / "update-check.json"


def _read_cache():
    """`(latest, age_seconds)` from the cache; either may be None.

    The two are independent on purpose. `age` records when PyPI was last *asked*,
    including the times it did not answer, so a laptop with no network backs off
    for a day like everyone else instead of paying the wait on every launch.
    `latest` is the last version actually learned, which may be older than the
    attempt or missing entirely.
    """
    try:
        with open(cache_path(), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        checked_at = float(payload.get("checked_at", 0))
    except (OSError, ValueError, TypeError):
        return None, None
    latest = payload.get("latest")
    age = max(0.0, time.time() - checked_at) if checked_at else None
    return (str(latest) if latest else None), age


def _write_cache(latest) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed: two sessions starting together
        # must not leave each other half a file to parse.
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"latest": latest, "checked_at": time.time()}, handle)
        os.replace(temporary, path)
    except OSError:
        pass  # A read-only home costs the notice, not the session.


def _fetch() -> None:
    """Refresh the cache. Runs on the background thread; never raises."""
    previous, _age = _read_cache()
    try:
        # Imported here rather than at module scope: this module is pulled in on
        # the path to the start screen, and `cli` is only cheap once something
        # else has already paid for it.
        from vtrace.cli import _fetch_latest_pypi_version

        latest = _fetch_latest_pypi_version(timeout=FETCH_TIMEOUT)
    except Exception:
        # RuntimeError for the reachability cases the fetcher names, but also
        # whatever a proxy layer or a patched urllib raises. None of it is worth
        # surfacing: the reader did not ask. The attempt is still recorded, so
        # an unreachable PyPI is asked once a day and not once a launch.
        _write_cache(previous)
        return
    with _lock:
        _result["latest"] = latest
    _write_cache(latest)


def start() -> None:
    """Begin a refresh if the cached answer is stale, and return immediately.

    Safe to call more than once; only the first stale call starts a thread.
    """
    global _thread
    if _thread is not None:
        return
    latest, age = _read_cache()
    if age is not None and age < CACHE_TTL:
        # Asked recently enough. Whatever it said then — a version, or nothing
        # because the network was down — stands until the cache goes stale.
        if latest is not None:
            with _lock:
                _result["latest"] = latest
        return
    _thread = threading.Thread(target=_fetch, name="vtrace-update-check", daemon=True)
    _thread.start()


def pending(current: str, wait: float = DEFAULT_WAIT):
    """The newer version on PyPI, or None when there is nothing to report.

    `wait` is how long to give a refresh that is still running. A fetch that
    misses the deadline is not cancelled — it finishes into the cache, and the
    next launch reads the answer straight off disk.
    """
    if not current:
        return None
    if _thread is not None and _thread.is_alive() and wait > 0:
        _thread.join(timeout=wait)
    with _lock:
        latest = _result.get("latest")
    if latest is None:
        latest, _age = _read_cache()
    if not latest:
        return None

    from vtrace.cli import _version_tuple

    # Strictly newer, by version order rather than inequality: a checkout built
    # ahead of the last release must not be told to update backwards.
    try:
        if _version_tuple(latest) <= _version_tuple(current):
            return None
    except Exception:
        return None
    return latest
