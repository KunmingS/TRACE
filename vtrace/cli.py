"""CLI entry point for TRACE.

Usage:
    vtrace app
    vtrace prepare
    vtrace train --model maev2b --pairs /my/data/video.mp4=/my/data/video.csv --output /my/runs
    vtrace eval --model-dir /my/runs/model_20260507_143012 --pairs /my/test/v.mp4=/my/test/v.csv
    vtrace predict --model-dir /my/runs/model_20260507_143012 --input /my/video.mp4
    vtrace update

A pair names its own files, so no folder has to be set first. Steps chain with
`then`, which replaces the old `vtrace pipeline`:

    vtrace > train --pairs /my/data/a.mp4=/my/data/a.csv --output /my/runs
             then eval --pairs /my/test/b.mp4=/my/test/b.csv
             then predict --input /my/new

See `vtrace.shell` for the prompt that makes a chain like that easy to type.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vtrace.config import DictAction
from vtrace.model_artifacts import (
    create_eval_dir,
    create_model_dir,
    resolve_model_dir,
)
from vtrace.resources import INPUT_RESOLUTIONS, RESOURCE_PROFILE_IDS
from vtrace.version import __version__
from vtrace.weights import model_weight_choices


# `--model NAME` -> config file. Adding a preset is one entry here plus the config;
# the argparse choices and the GUI's command builder both read this dict.
MODEL_CONFIGS = {
    "maev2b": "configs/maev2b.py",
    "maev2b-distilled": "configs/maev2b_distilled.py",
    "vjepa2": "configs/vjepa2.py",
}
DEFAULT_MODEL = next(iter(MODEL_CONFIGS))
PYPI_PROJECT_NAME = "vtrace-behavior"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PROJECT_NAME}/json"


def _require_cuda():
    """Exit with a clear message if no CUDA-capable GPU is available."""
    import torch
    if not torch.cuda.is_available():
        print(
            "Error: TRACE requires a CUDA-capable GPU.\n"
            "  torch.cuda.is_available() returned False.\n"
            "  Check: nvidia-smi, your PyTorch CUDA build, and CUDA_VISIBLE_DEVICES.",
            file=sys.stderr,
        )
        sys.exit(2)


def _resolve_config(args):
    """Resolve config path from --model or --config."""
    if getattr(args, "config", None):
        return args.config
    model = getattr(args, "model", DEFAULT_MODEL)
    try:
        rel = MODEL_CONFIGS[model]
    except KeyError:
        raise SystemExit(
            f"Unknown --model {model!r}; choose from {', '.join(MODEL_CONFIGS)} "
            f"or pass --config with an explicit config file."
        )
    # Try relative to package root first, then cwd
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_path = os.path.join(pkg_dir, rel)
    if os.path.isfile(pkg_path):
        return pkg_path
    cwd_path = os.path.join(os.getcwd(), rel)
    if os.path.isfile(cwd_path):
        return cwd_path
    # Fall back to relative (let downstream error handle it)
    return rel


def _write_prep_result(model_dir, dataset_json, classmap_path):
    result = {
        "model_dir": model_dir,
        "dataset_json": dataset_json,
        "classmap_path": classmap_path,
    }
    with open(os.path.join(model_dir, "prep_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _prepare_pairs_into(output_dir, pairs, *, subset,
                        proxy_geometry=None, flag="--pairs"):
    """Prepare explicit video/CSV pairs into ``output_dir`` as one subset.

    ``subset`` is `train` or `validation`, and it is the caller's decision:
    training data and evaluation data are different corpora chosen separately,
    so there is nothing here to split.

    A pair names its own files, so there is no folder argument. Relative paths
    resolve against the working directory, which is where the reader is standing
    and what every other command-line tool would do with them.
    """
    from vtrace.data_prep import DEFAULT_PROXY_GEOMETRY, prepare_dataset

    if not pairs:
        print(f"Error: {flag} is required. Pass each video/CSV as VIDEO_PATH=CSV_PATH.")
        sys.exit(1)

    print(f"Preparing {len(pairs)} pair(s) as `{subset}`:")
    for spec in pairs:
        print(f"  {spec}")
    if proxy_geometry is None:
        proxy_geometry = DEFAULT_PROXY_GEOMETRY
    try:
        output_dir, dataset_json, classmap_path = prepare_dataset(
            os.getcwd(),
            subset=subset,
            proxy_geometry=proxy_geometry,
            explicit_pairs=pairs,
            output_dir=output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        # A mistyped path is the likeliest first error anyone makes, and every
        # one of these already says exactly what is wrong. A traceback on top of
        # that only buries it.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    _write_prep_result(output_dir, dataset_json, classmap_path)
    print()
    return output_dir, dataset_json, classmap_path


def _merged_config(config_path, resolution=None):
    """The config a run will really use, with `--input-resolution` folded in.

    Read here rather than left to the training subprocess because prep needs it
    first: the decode proxies are built to this config's geometry, and a proxy
    built at another resolution would have training decode upsampled frames.
    """
    from vtrace.config import Config
    from vtrace.proxy_geometry import input_resize_cfg_options

    cfg = Config.fromfile(config_path)
    if resolution:
        cfg.merge_from_dict(input_resize_cfg_options(cfg, resolution))
    return cfg


def _resource_cfg_options(profile_id, cfg, *, training, resolution=None):
    """Dataloader overrides for the chosen profile, or none when unset.

    Unset means the config decides. A profile is a deliberate override of what
    the config author picked, so silently applying one by default would make
    every run ignore its own solver block.
    """
    if not profile_id:
        return {}
    from vtrace.resources import eval_cfg_options, profile_by_id, train_cfg_options

    profile = profile_by_id(profile_id)
    if training:
        return train_cfg_options(profile, cfg, resolution)
    return eval_cfg_options(profile, cfg)


def _model_info_or_exit(model_dir):
    try:
        return resolve_model_dir(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def _fetch_latest_pypi_version(timeout=5.0, url=PYPI_JSON_URL):
    """Return the latest version published on PyPI."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # No such project. Distinct from a failed check: there is nothing to
            # update to, which is the normal state before the first release.
            raise RuntimeError(
                f"{PYPI_PROJECT_NAME} is not published on PyPI yet."
            ) from exc
        raise RuntimeError(f"PyPI returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Could not reach PyPI: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Timed out while checking PyPI") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("PyPI returned invalid JSON") from exc

    latest = payload.get("info", {}).get("version")
    if not latest:
        raise RuntimeError("PyPI response did not include a version")
    return str(latest)


def gui_path():
    """The V-TRACE GUI shipped inside the package."""
    candidate = Path(__file__).resolve().parent / "static" / "gui" / "index.html"
    return candidate if candidate.is_file() else None


# ── folders named, not located ───────────────────────────────────────────────
# The configuration page picks folders through the browser, and the browser
# tells it what a folder is called and nothing about where it is: the File
# System Access API hands the page a capability, never a location, because an
# absolute path leaks the user's name and layout. So the page writes the NAME
# into the command, with a fingerprint of the folder's contents after it —
# `--output '<runs#3f9a2c1e>'`, `--pairs '<videos#8b1d0c47>/a.mp4=…'` — and
# the lookup happens here, on the machine that runs the command.
#
# That machine is the only one whose paths matter. A folder picked on a laptop
# through a network mount has one path there and another on the GPU box the
# command is pasted into; what the two share is the folder's name and the files
# inside it, which is exactly what the token carries. Resolving in the page
# would have written the laptop's path into a command meant for the server.
#
# Only the interactive session resolves. A one-shot `vtrace train …` from a
# shell or a job script refuses a token and says what to replace it with: a job
# on a cluster runs unattended, where a lookup that guesses or asks is worse
# than one that never starts.
_FOLDER_TOKEN = re.compile(r"<([^<>/]+)>")
_FINGERPRINT_MARK = "#"
# Written by the page into a folder it may write to and has nothing to
# fingerprint — a fresh output folder — so that folder can still be told apart
# from another of the same name. Eight hex digits, same width as a fingerprint.
MARKER_FILE = ".vtrace-folder"
# What the page writes when no folder was picked at all. Nothing to look up.
_EMPTY_TOKENS = {"video folder", "output folder", "eval folder", "model folder",
                 "video.mp4", "video.csv"}
# Where `train` puts its run folder when `--output` is not given: under the
# working directory the session was started in. The page leaves the flag out
# when no output folder was picked, so the one folder a person had to choose by
# hand is now chosen for them — and the session's own ground, not the videos',
# is where it lands. Kept in step with the configuration page's output card.
DEFAULT_OUTPUT_DIR = "runs"
# Kept in step with VIDEO_EXTENSIONS in the configuration page.
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
# Extra places to look, for a machine whose data lives nowhere near the working
# directory or home: `VTRACE_ROOTS=/data:/mnt/lab vtrace`.
_ROOTS_ENV = "VTRACE_ROOTS"
_SCAN_DEPTH = 3
_SCAN_LIMIT = 4000
_SCAN_SKIP = {".git", "__pycache__", "node_modules", "venv", ".venv", "site-packages"}
# What preparation leaves beside a video (see data_prep.SIDECAR_SUFFIX): a cache,
# never the folder someone is looking for, and there is one per video.
_SIDECAR_SUFFIX = ".vtrace"


def default_output_dir() -> str:
    """The folder `train` creates its run in when no `--output` was given."""
    return os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)


def folder_fingerprint(path) -> str:
    """Eight hex digits that identify a folder by what is in it, not where it is.

    The rule the page applies to the folder it picked, so the two agree: the
    sorted names of the videos in the folder when there are any, otherwise of
    every visible entry (a run folder has no videos, but it has best.pth and
    classmap.txt), SHA-256'd and cut to eight digits. Videos only, when
    possible, because that is the list that stays put — preparation drops a
    `<video>.vtrace` folder beside each video, and the CSVs beside them are
    edited, so a fingerprint over everything would break between
    copying the command and running it. Hidden entries are skipped on both
    sides (.DS_Store, AppleDouble sidecars, the marker file itself).

    Empty string when the folder cannot be read or has nothing visible in it.
    """
    import hashlib
    import unicodedata

    try:
        entries = [entry for entry in os.scandir(path) if not entry.name.startswith(".")]
    except OSError:
        return ""
    # NFC on both sides: a name that arrives decomposed from a macOS mount must
    # hash the same as the composed one the server's own disk reports.
    names = [unicodedata.normalize("NFC", entry.name) for entry in entries]
    videos = [name for name, entry in zip(names, entries)
              if entry.is_file() and name.lower().endswith(_VIDEO_SUFFIXES)]
    chosen = sorted(videos or names)
    if not chosen:
        return ""
    return hashlib.sha256("\n".join(chosen).encode("utf-8")).hexdigest()[:8]


def folder_marker(path) -> str:
    """The id in a folder's marker file, or empty when there is none."""
    try:
        with open(Path(path) / MARKER_FILE, encoding="utf-8") as handle:
            return handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _fingerprint_matches(path, fingerprint: str) -> bool:
    return fingerprint in (folder_fingerprint(path), folder_marker(path))


def _scan_roots() -> list[Path]:
    """Where to look for the folder the page named.

    A data folder beside the checkout is as common as one inside it, and neither is
    reachable from the other by walking down, so the parent and home are searched
    too, plus anything named in VTRACE_ROOTS. Ordered by relevance: the entry
    budget is spent in order.
    """
    cwd = Path.cwd()
    extra = [Path(part) for part in os.environ.get(_ROOTS_ENV, "").split(os.pathsep) if part]
    roots: list[Path] = []
    for candidate in (cwd, cwd.parent, Path.home(), *extra):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def _local_directories(root, limit: int = _SCAN_LIMIT) -> dict:
    """basename -> [absolute paths], for directories at or under `root`.

    `root` may be one path or several; the walk is breadth-first across all of
    them, so every root's immediate children are indexed before anything deeper
    anywhere. That is what the entry budget needs to be spent on: a data folder
    sitting directly beside the checkout is a likelier pick than the fifth level
    of a tree inside it, and depth-first ordering would spend the whole budget
    on the latter before ever reaching the former.
    """
    from collections import deque

    index: dict[str, list[str]] = {}
    seen = 0
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    queue = deque((Path(r), 0) for r in roots)
    walked: set[str] = set()
    while queue and seen < limit:
        current, depth = queue.popleft()
        # Roots overlap by construction (cwd sits under its own parent), so the same
        # directory would otherwise be walked, and indexed, more than once.
        key = str(current)
        if key in walked:
            continue
        walked.add(key)
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if seen >= limit:
                break
            if not entry.is_dir(follow_symlinks=False):
                continue
            if (entry.name.startswith(".") or entry.name in _SCAN_SKIP
                    or entry.name.endswith(_SIDECAR_SUFFIX)):
                continue
            paths = index.setdefault(entry.name, [])
            if entry.path not in paths:
                paths.append(entry.path)
            seen += 1
            if depth + 1 < _SCAN_DEPTH:
                queue.append((Path(entry.path), depth + 1))
    return index


def _remembered_file() -> Path:
    from vtrace.splash import trace_home

    return trace_home() / "folders.json"


def _remembered_folders() -> list[str]:
    """Folders earlier commands resolved to. A name found once is found again.

    The scan is three levels under a few roots, and a data folder on a big
    disk is often deeper than that or somewhere else entirely. Once it has been
    found — or typed as a full path the first time — its path is kept, so the
    next command that names it does not depend on where the session was started.
    """
    try:
        data = json.loads(_remembered_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [path for path in data if isinstance(path, str)] if isinstance(data, list) else []


def _remember_folders(paths) -> None:
    known = _remembered_folders()
    new = [path for path in paths if path not in known]
    if not new:
        return
    target = _remembered_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(known + new, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _directory_index() -> dict:
    """basename -> [paths] for every folder a named token could mean.

    The scan roots, plus the folders earlier commands resolved to, plus the
    demo's own folders, which live under ~/.vtrace and would otherwise be hidden
    by the dot-directory rule.
    """
    from vtrace.demo import demo_dir

    index = _local_directories(_scan_roots())

    def add(path: Path) -> None:
        if path.is_dir():
            paths = index.setdefault(path.name, [])
            if str(path) not in paths:
                paths.append(str(path))

    for remembered in _remembered_folders():
        add(Path(remembered))
    demo = demo_dir()
    if demo.is_dir():
        for extra in (demo, demo / "videos" / "train", demo / "videos" / "test"):
            add(extra)
    return index


def _split_token(inner: str):
    """`videos#8b1d0c47` -> ('videos', '8b1d0c47'); `videos` -> ('videos', '')."""
    name, _mark, fingerprint = inner.partition(_FINGERPRINT_MARK)
    return name, fingerprint


def _replace_instructions(inners) -> str:
    """What a shell or a job script is told instead of a lookup."""
    lines = [
        "This command names folders instead of locating them, so it cannot run",
        "as it is here. The `vtrace` session finds them — start it and paste the",
        "command into the box it opens. Anywhere else, replace each token with",
        "the folder's full path on the machine that runs the command:",
    ]
    width = max(len(inner) for inner in inners) + 2
    for inner in inners:
        name, _fingerprint = _split_token(inner)
        lines.append(f"  <{inner}>{' ' * (width - len(inner))}the folder called \"{name}\" "
                     f"that was picked in the page")
    return "\n".join(lines)


def resolve_placeholders(argv):
    """Replace `<folder-name>` / `<folder-name#fingerprint>` tokens in `argv`.

    Session only: anywhere else a token is an error that names what to replace
    it with. In the session, a name is looked up three levels under the working
    directory, its parent, home and VTRACE_ROOTS, plus every folder resolved
    before. A fingerprint keeps only the candidates whose contents match the
    folder the page saw, so a same-named folder somewhere else is never taken.
    Several survivors are settled by the one under the working directory if
    there is exactly one there, else by asking — `--output` says where a run
    gets written, and a confident wrong path is worse than a question.
    """
    from vtrace import splash
    from vtrace.console import ACCENT, say

    inners = []
    for token in argv:
        for inner in _FOLDER_TOKEN.findall(token):
            if inner not in inners:
                inners.append(inner)
    if not inners:
        return argv

    empty = [inner for inner in inners if inner in _EMPTY_TOKENS]
    if empty:
        raise SystemExit(
            f"The command still has <{empty[0]}>: the page had no folder picked "
            f"for it. Pick one there, or replace it with a path.")

    if not splash.IN_SESSION:
        raise SystemExit(_replace_instructions(inners))

    cwd = str(Path.cwd())
    index = _directory_index()
    resolved = {}
    for inner in inners:
        name, fingerprint = _split_token(inner)
        matches = index.get(name, [])
        if fingerprint and matches:
            verified = [path for path in matches if _fingerprint_matches(path, fingerprint)]
            if not verified:
                listed = "\n".join(f"  {path}" for path in matches)
                raise SystemExit(
                    f"{len(matches)} folder{'s' if len(matches) > 1 else ''} called "
                    f"\"{name}\" found, but none holds the files the page saw when "
                    f"it was picked:\n{listed}\n"
                    f"If it is a different folder, run this from nearer to it, set "
                    f"{_ROOTS_ENV}, or replace <{inner}> with its full path. If the "
                    f"folder's contents changed, copy the command from the page again.")
            matches = verified
        if len(matches) > 1:
            inside = [path for path in matches if path.startswith(cwd + os.sep)]
            if len(inside) == 1:
                matches = inside
        if len(matches) == 1:
            resolved[inner] = matches[0]
        elif not matches:
            raise SystemExit(
                f"No folder called \"{name}\" within three levels of {cwd}, its "
                f"parent, or your home folder. Run this from nearer the data, set "
                f"{_ROOTS_ENV}=/where/it/lives, or replace <{inner}> with the full path.")
        elif sys.stdin.isatty() and sys.stdout.isatty():
            resolved[inner] = _ask_which(name, matches)
        else:
            listed = "\n".join(f"  {path}" for path in matches)
            raise SystemExit(
                f"{len(matches)} folders are called \"{name}\":\n{listed}\n"
                f"Replace <{inner}> with the one you mean.")
        say(f"  [dim]<{inner}>[/] → [{ACCENT}]{resolved[inner]}[/]")
    _remember_folders(resolved.values())

    def substitute(token):
        return _FOLDER_TOKEN.sub(lambda m: resolved.get(m.group(1), m.group(0)), token)

    return [substitute(token) for token in argv]


def _ask_which(name, matches):
    """One question, at the prompt, when a name fits several folders."""
    from vtrace.console import ACCENT, say

    say(f"\n  [bold]{len(matches)}[/] folders are called [{ACCENT}]{name}[/]:")
    for number, path in enumerate(matches, 1):
        say(f"    [{ACCENT}]{number}[/]  {path}")
    while True:
        try:
            answer = input(f"  which one? [1-{len(matches)}] ").strip()
        except EOFError:
            raise SystemExit(f"Replace <{name}> with the folder you mean.")
        if answer.isdigit() and 1 <= int(answer) <= len(matches):
            return matches[int(answer) - 1]
        say("  [dim]a number from the list, please[/]")


def _local_context() -> str:
    """The `window.__VTRACE__` script tag injected into the served page.

    Only how the page was opened. It used to carry a directory index so the page
    could write real paths into the command; the page now writes names and
    fingerprints and the session does the lookup, on the machine that runs the
    command — see the note above `resolve_placeholders`.
    """
    payload = {"served": True, "cwd": str(Path.cwd())}
    # `</script>` inside the JSON would end the tag early; nothing else can escape it.
    encoded = json.dumps(payload).replace("</", "<\\/")
    return f"<script>window.__VTRACE__ = {encoded};</script>"


class _GuiHandler(SimpleHTTPRequestHandler):
    """Serves the single GUI file, and nothing else on disk.

    Every path maps to that one file: the GUI routes itself with `?page=`, so
    there is nothing else to hand out, and a directory-listing server rooted at
    the user's filesystem is not something to expose even on localhost.
    """

    gui_file = None
    quiet = True
    # Filled in by `start_gui_server`: the `window.__VTRACE__` tag that tells
    # the page it was served rather than opened as a file.
    local_context = None

    def do_GET(self):
        # `?page=` lives in the query string, so the only real path is "/". Anything
        # else is a 404 — including the `/api/status` probe the configuration page
        # makes, whose failure is exactly what puts it in browser-only mode.
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            self.send_error(404)
            return
        try:
            body = self.gui_file.read_bytes()
        except OSError as exc:
            self.send_error(500, explain=str(exc))
            return
        if self.local_context:
            # Before the page's own script, so the global exists when it runs.
            body = body.replace(b"</head>", self.local_context.encode("utf-8") + b"</head>", 1)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # The file is replaced in place by an upgrade, so never let it be cached.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_HEAD = do_GET

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


# The only interface worth binding — see `serve` for why a LAN bind would serve
# strictly less than double-clicking the file.
LOOPBACK = "127.0.0.1"
DEFAULT_GUI_PORT = 8765
# Ports past the default to try before giving up. A stale server from an earlier
# session should not turn `vtrace app` into a puzzle.
_PORT_SEARCH_SPAN = 20


def _bind_first_free(handler):
    """Bind the first free port at or after `DEFAULT_GUI_PORT`; (None, None) if none."""
    for port in range(DEFAULT_GUI_PORT, DEFAULT_GUI_PORT + _PORT_SEARCH_SPAN):
        try:
            return ThreadingHTTPServer((LOOPBACK, port), handler), port
        except OSError:
            continue
    return None, None


def start_gui_server(port=None, verbose=False):
    """Bind the annotator server and return (httpd, port), or (None, None).

    Returns rather than serves, so the caller decides whether to block on it or
    run it on a thread.
    """
    gui = gui_path()
    if gui is None:
        return None, None
    handler = type("GuiHandler", (_GuiHandler,), {
        "gui_file": gui,
        "quiet": not verbose,
        "local_context": _local_context(),
    })
    if port is not None:
        try:
            return ThreadingHTTPServer((LOOPBACK, port), handler), port
        except OSError:
            return None, None
    return _bind_first_free(handler)


def serve_in_background():
    """Start the annotator on a daemon thread; return its URL, or None.

    The interactive session brings the annotator up by itself: it is the one thing
    every user needs and there is nothing to decide about it, so making someone type
    `vtrace app` first is a step that only exists to be skipped. A daemon thread dies
    with the process, so leaving the session needs no teardown.
    """
    httpd, port = start_gui_server()
    if httpd is None:
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://localhost:{port}"


def serve(args):
    """Serve the V-TRACE GUI over http://localhost.

    Not for the browser APIs: a `file://` page is a secure context in current
    Chrome, and gets IndexedDB and the File System Access API just the same.
    One thing it does not get: every `file://` page on the machine shares one
    origin, so a second copy of this file — or any other local page — reads the
    same IndexedDB, down to the stored directory handles. `http://localhost:PORT`
    gives the page an origin of its own. (Paths are not the difference: the page
    writes folder names and fingerprints either way, and the session that runs
    the command looks them up — see `resolve_placeholders`.)

    Loopback only. There is nothing a LAN bind would add: the page reads videos
    from the computer running the browser, and the command it writes is run
    wherever it is pasted.
    """
    if gui_path() is None:
        print("Error: the GUI file is missing from this installation "
              "(vtrace/static/gui/index.html).", file=sys.stderr)
        sys.exit(1)

    httpd, port = start_gui_server(args.port, args.verbose)
    if httpd is None:
        if args.port is not None:
            # Explicitly asked for: fail loudly rather than quietly using another port.
            print(f"\n  Cannot serve on {LOOPBACK}:{args.port} — port unavailable.\n",
                  file=sys.stderr)
        else:
            print(f"\n  Every port from {DEFAULT_GUI_PORT} to "
                  f"{DEFAULT_GUI_PORT + _PORT_SEARCH_SPAN - 1} is in use.\n"
                  f"  Free one, or pick another: vtrace app --port N\n", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{port}"
    print()
    print("  V-TRACE annotator")
    print()
    print(f"  {url}")
    print()
    print("  Videos are read straight from this computer — nothing is uploaded.")
    print("  Press Ctrl+C to stop.")
    print()

    if not args.no_browser:
        # Opened here rather than printed for the reader to click: `vtrace app` has
        # exactly one thing to do next, so doing it is better than instructing it.
        if webbrowser.open(url):
            print(f"  Opening {url} ...")
        else:
            print(f"  Could not open a browser — go to {url}")
        print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _download_weights_selection(selection):
    """Download the selected model weights and print their local paths."""
    from vtrace.weights import download_model_weights

    paths = download_model_weights(selection)
    for path in paths:
        print(path)
    return paths


def _ensure_ffmpeg() -> None:
    """Download bundled ffmpeg/ffprobe via static_ffmpeg if not on PATH.

    Called from `prepare` and `update` so a fresh install / upgrade lands
    with working video tooling on Windows (where ffmpeg isn't part of the OS).
    """
    have_ffmpeg = shutil.which("ffmpeg")
    have_ffprobe = shutil.which("ffprobe")
    if have_ffmpeg and have_ffprobe:
        print(f"ffmpeg already available ({have_ffmpeg}).")
        return
    try:
        import static_ffmpeg
        from static_ffmpeg.run import get_platform_dir
    except ImportError:
        print("Warning: static-ffmpeg not installed; cannot auto-fetch ffmpeg.",
              file=sys.stderr)
        return

    # Distinguish "first-time fetch" from "binaries already cached, just
    # activate them on PATH" so the ~80MB message doesn't lie on every run.
    cache_dir = get_platform_dir()
    cache_ready = (
        os.path.isfile(os.path.join(cache_dir, "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"))
        and os.path.isfile(os.path.join(cache_dir, "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"))
    )
    if cache_ready:
        print("Activating bundled ffmpeg/ffprobe...")
    else:
        print("Fetching bundled ffmpeg/ffprobe (one-time, ~80MB)...")

    try:
        static_ffmpeg.add_paths(weak=True)
    except Exception as exc:
        print(f"Warning: ffmpeg fetch failed: {exc}", file=sys.stderr)
        return
    print(f"  ffmpeg:  {shutil.which('ffmpeg')}")
    print(f"  ffprobe: {shutil.which('ffprobe')}")


def prepare(args):
    """Prepare local assets needed by TRACE."""
    _ensure_ffmpeg()

    if args.weights == "none":
        print("No weight downloads selected.")
        return []

    print(f"Preparing model weights: {args.weights}")
    paths = _download_weights_selection(args.weights)
    print()
    print("TRACE is ready.")
    return paths


def _version_tuple(version):
    """(1, 0, 0) from "1.0.0", for ordering. Trailing non-numeric parts are dropped."""
    parts = []
    for chunk in str(version).split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _run_pip_upgrade():
    """Upgrade the installed V-TRACE package via pip. Returns pip's exit code."""
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PROJECT_NAME]
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def update(args):
    """Report whether a newer V-TRACE is on PyPI, and offer to install it."""
    try:
        latest = _fetch_latest_pypi_version(timeout=args.timeout)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    current = __version__
    return_code = 0
    # Compare by version order, not equality: a checkout ahead of PyPI (a release
    # candidate, or a local build) must not be told to "update" backwards.
    if _version_tuple(latest) <= _version_tuple(current):
        if latest == current:
            print(f"V-TRACE is up to date ({current}).")
        else:
            print(f"V-TRACE {current} is installed; PyPI has {latest}. Nothing to update.")
    else:
        print(f"V-TRACE {latest} is available on PyPI.")
        print(f"Installed version: {current}")
        print()

        # --yes auto-confirms and --check-only never installs; otherwise ask, but
        # only at a terminal, so a piped or scheduled run never hangs on input.
        if getattr(args, "check_only", False):
            do_install = False
        elif getattr(args, "yes", False):
            do_install = True
        elif sys.stdin is not None and sys.stdin.isatty():
            try:
                answer = input(f"Install {PYPI_PROJECT_NAME} {latest} now? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            do_install = answer in ("y", "yes")
        else:
            do_install = False

        if do_install:
            return_code = _run_pip_upgrade()
            if return_code == 0:
                print(f"\nUpdated to {PYPI_PROJECT_NAME} {latest}. "
                      "Restart any running V-TRACE processes to use it.")
            else:
                print(f"\nUpdate failed (pip exit {return_code}). Run it manually:")
                print(f"  python -m pip install --upgrade {PYPI_PROJECT_NAME}")
        else:
            print("Update with:")
            print(f"  python -m pip install --upgrade {PYPI_PROJECT_NAME}")

    # Upgrades can introduce new native deps. Run after the version check so
    # the user sees their V-TRACE status first, then any ffmpeg work.
    print()
    _ensure_ffmpeg()
    return return_code


def _tuned_profile_or_exit(config_path, model_dir, annotation_path, class_map):
    """Benchmark the training dataloader and return the profile it recommends."""
    from vtrace.resources import profile_by_name
    from vtrace.steps import TrainTuneRequest, run_train_tune

    print("\n--- Tuning train resources ---")
    step = run_train_tune(TrainTuneRequest(
        config_path=config_path,
        model_dir=model_dir,
        annotation_path=annotation_path,
        class_map=class_map,
    ))
    if not step.ok:
        print(f"\nResource tuning failed (exit code {step.returncode}); see {step.log_file}")
        sys.exit(step.returncode)
    with open(os.path.join(step.work_dir, "train_tune_result.json"), encoding="utf-8") as f:
        recommended = json.load(f).get("recommended_profile")
    profile = profile_by_name(recommended)
    print(f"Using the `{profile.id}` resource profile.\n")
    return profile.id


def train(args):
    """Train a model, and score it each epoch when evaluation data is given."""
    _require_cuda()
    from vtrace.data_prep import TRAIN_SUBSET, VALIDATION_SUBSET
    from vtrace.proxy_geometry import config_geometry
    from vtrace.steps import TrainRequest, run_train

    config_path = _resolve_config(args)
    cfg = _merged_config(config_path, args.input_resolution)
    geometry = config_geometry(cfg)

    # No --output: the run goes under the working directory, and says so before
    # anything is written, so the path is on screen when the person looks for it.
    output = args.output or default_output_dir()
    if not args.output:
        print(f"No --output given, so the run folder goes under {output}")
    model_dir = create_model_dir(output)
    model_dir, dataset_json, classmap_path = _prepare_pairs_into(
        model_dir, args.explicit_pairs,
        subset=TRAIN_SUBSET, proxy_geometry=geometry,
    )

    # Evaluation data is a corpus of its own, prepared into its own folder under
    # the run. Nothing is held back from the training videos: which videos score
    # the model is a choice, and it belongs to whoever knows the recordings.
    eval_annotation = eval_data_dir = None
    if args.eval_pairs:
        eval_data_dir, eval_annotation, _ = _prepare_pairs_into(
            os.path.join(model_dir, "eval_data"),
            args.eval_pairs,
            subset=VALIDATION_SUBSET, proxy_geometry=geometry, flag="--eval-pairs",
        )

    profile_id = args.resource_profile
    if profile_id == "auto":
        profile_id = _tuned_profile_or_exit(
            config_path, model_dir, dataset_json, classmap_path
        )

    # Epoch counts stay unset unless asked for, so a config's own schedule is
    # what runs. An override has to reach both keys: `scheduler.max_epoch`
    # shapes the LR curve and `workflow.end_epoch` stops the loop.
    cfg_options = {}
    if args.epochs is not None:
        cfg_options["scheduler.max_epoch"] = args.epochs
        cfg_options["workflow.end_epoch"] = args.epochs
    if args.val_start_epoch is not None:
        cfg_options["workflow.val_start_epoch"] = args.val_start_epoch
    if args.val_interval is not None:
        cfg_options["workflow.val_eval_interval"] = args.val_interval
    cfg_options.update(_resource_cfg_options(
        profile_id, cfg, training=True, resolution=args.input_resolution
    ))
    if args.cfg_options:
        cfg_options.update(args.cfg_options)

    request = TrainRequest(
        config_path=config_path,
        model_dir=model_dir,
        seed=args.seed,
        resume=args.resume,
        not_eval=args.not_eval,
        disable_deterministic=args.disable_deterministic,
        dataset_dir=model_dir,
        annotation_path=dataset_json,
        class_map=classmap_path,
        eval_annotation_path=eval_annotation,
        eval_data_dir=eval_data_dir,
        pretrained=args.pretrained,
        cfg_options=cfg_options or None,
    )

    result = run_train(request)

    if not result.ok:
        print(f"\nTraining failed (exit code {result.returncode}); see {result.log_file}")
        sys.exit(result.returncode)
    print("\nTraining completed successfully.")
    print(f"Model directory: {model_dir}")
    if not eval_annotation:
        print("  No evaluation data, so no epoch was scored: the folder holds "
              "last.pth, not best.pth.")
    return model_dir


def test(args):
    """Evaluate a finished model on annotated videos."""
    _require_cuda()
    from vtrace.data_prep import VALIDATION_SUBSET
    from vtrace.proxy_geometry import config_geometry
    from vtrace.steps import TestRequest, run_test

    model_info = _model_info_or_exit(args.model_dir)
    # Without `--pairs`, the data is the corpus the run scored its own epochs
    # against. NOT the run's `dataset.json`: that one holds the training videos,
    # every entry labelled `train`, and an evaluation pass reading it would find
    # nothing to score.
    annotation_path = model_info["eval_dataset_json"]
    dataset_dir = (
        os.path.dirname(annotation_path) if annotation_path else model_info["model_dir"]
    )
    output_dir = create_eval_dir(model_info["model_dir"])

    # The geometry comes from the model's own resolved config, not from a
    # default: proxies built to another resolution would feed this model
    # upsampled frames and quietly cost it accuracy.
    cfg = _merged_config(model_info["config_path"])
    if args.explicit_pairs:
        dataset_dir, annotation_path, _ = _prepare_pairs_into(
            output_dir, args.explicit_pairs,
            subset=VALIDATION_SUBSET, proxy_geometry=config_geometry(cfg),
        )
    elif not annotation_path:
        print("Error: this run has no evaluation data of its own — it was trained "
              "without --eval-pairs. Pass --pairs naming the videos to score it on.",
              file=sys.stderr)
        sys.exit(1)

    cfg_options = dict(_resource_cfg_options(args.resource_profile, cfg, training=False))
    if args.cfg_options:
        cfg_options.update(args.cfg_options)

    request = TestRequest(
        model_dir=model_info["model_dir"],
        output_dir=output_dir,
        seed=args.seed,
        not_eval=args.not_eval,
        profile=args.profile,
        auto_tune=args.auto_tune,
        dataset_dir=dataset_dir,
        annotation_path=annotation_path,
        cfg_options=cfg_options or None,
    )

    result = run_test(request)

    if not result.ok:
        print(f"\nTesting failed (exit code {result.returncode}); see {result.log_file}")
        sys.exit(result.returncode)
    print("\nTesting completed successfully.")
    return model_info["model_dir"]


def infer(args):
    """Run prediction on videos that need no annotations."""
    _require_cuda()
    from vtrace.steps import InferRequest, run_infer

    model_info = _model_info_or_exit(args.model_dir)
    cfg_options = dict(_resource_cfg_options(
        args.resource_profile, _merged_config(model_info["config_path"]), training=False
    ))
    if args.cfg_options:
        cfg_options.update(args.cfg_options)

    request = InferRequest(
        model_dir=model_info["model_dir"],
        input=args.input,
        output=args.output,
        seed=args.seed,
        profile=args.profile,
        auto_tune=args.auto_tune,
        threshold=args.threshold,
        included_stems=args.include_stems or None,
        cfg_options=cfg_options or None,
    )

    result = run_infer(request)

    if not result.ok:
        print(f"\nInference failed (exit code {result.returncode}); see {result.log_file}")
        sys.exit(result.returncode)
    print("\nInference completed successfully.")
    return model_info["model_dir"]


def demo(args):
    """Dispatch `vtrace demo <step>`."""
    from vtrace import demo as demo_mod

    step = {"download": demo_mod.download, "predict": demo_mod.predict, "train": demo_mod.train}
    return step[args.demo_command](args)


def demo_default_jobs():
    """The demo's default download concurrency, for `--help`.

    Imported here rather than at module scope so building the parser stays as
    cheap as it is for every other subcommand.
    """
    from vtrace.demo import DEFAULT_JOBS

    return DEFAULT_JOBS


def _add_serve_args(parser):
    parser.add_argument("--port", type=int, default=None,
        help=f"Port (default: {DEFAULT_GUI_PORT}, or the next free one)")
    parser.add_argument("--no-browser", action="store_true",
        help="Do not open a browser window")
    parser.add_argument("--verbose", action="store_true",
        help="Log every request to this terminal")
    parser.set_defaults(func=serve)


def _add_model_config_args(parser):
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS), default=DEFAULT_MODEL,
        help=f"Model preset (default: {DEFAULT_MODEL})")
    parser.add_argument("--config", type=str, default=None,
        help="Custom config file path (overrides --model)")


def _add_model_dir_arg(parser):
    parser.add_argument("--model-dir", type=str, required=True,
        help="Model artifact directory produced by `vtrace train`")


def _add_pair_args(parser, *, required=True, flag="--pairs", label="training"):
    parser.add_argument(flag, dest="explicit_pairs" if flag == "--pairs" else "eval_pairs",
        nargs="+", required=required, metavar="VIDEO=CSV",
        help=f"The {label} videos and their annotation CSVs, as VIDEO_PATH=CSV_PATH. "
             "Full paths; relative ones resolve against the working directory. "
             "Each pair names its own files, so there is no folder to set first.")


def _add_resource_profile_arg(parser, *, include_auto=False):
    choices = [*RESOURCE_PROFILE_IDS, "auto"] if include_auto else list(RESOURCE_PROFILE_IDS)
    extra = (" `auto` benchmarks the training dataloader first."
             if include_auto else "")
    parser.add_argument("--resource-profile", choices=choices, default=None,
        help="Dataloader batch size, workers, decode threads and prefetch depth "
             "as one named setting. Default: whatever the model config asks for."
             + extra + " Use --cfg-options for finer control.")


def _add_common_job_args(parser, *, include_profile=False, include_auto_tune=False):
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    if include_profile:
        parser.add_argument("--profile", action="store_true",
            help="Enable inference profiling (CPU + GPU timing breakdown)")
    if include_auto_tune:
        parser.add_argument("--auto-tune", action=argparse.BooleanOptionalAction, default=False,
            help="Run benchmark-based dataloader tuning (default: disabled)")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
        help="Override config settings (key=value pairs)")


def _add_train_args(parser):
    _add_model_config_args(parser)
    _add_pair_args(parser)
    # Optional, but never derived from where the videos happen to live. A corpus
    # is an input that many runs read; which of them writes its checkpoints
    # beside it is a decision, and an unasked one would put run folders wherever
    # the data sits. Left out, the run goes to `runs/` under the working
    # directory instead — the session's own ground.
    parser.add_argument("--output", type=str, default=None,
        help="Directory to create the run folder in (default: "
             f"{DEFAULT_OUTPUT_DIR}/ under the working directory). The run writes "
             "model_YYYYMMDD_HHMMSS/ here, holding the checkpoints, the class "
             "map and the resolved config.")
    # Evaluation data turns a training run from "every epoch was saved" into
    # "one epoch was chosen". Separate videos, not a slice of the training ones.
    parser.add_argument("--eval-pairs", nargs="+", default=None, metavar="VIDEO=CSV",
        help="Evaluation videos and their CSVs, as VIDEO_PATH=CSV_PATH. Given "
             "these, the training loop scores the model on them and writes "
             "best.pth; without them the run keeps every epoch's checkpoint and "
             "names none of them best. Equivalent to `train ... then eval ...`.")
    parser.add_argument("--pretrained", type=str, default=None,
        help="Pretrained backbone weights path (overrides config's pretrain)")
    parser.add_argument("--epochs", type=int, default=None,
        help="Total training epochs (default: the config's own schedule)")
    parser.add_argument("--val-start-epoch", type=int, default=None,
        help="First epoch to score against the evaluation data (default: config)")
    parser.add_argument("--val-interval", type=int, default=None,
        help="Epochs between evaluation passes (default: config)")
    parser.add_argument("--input-resolution", type=int, choices=list(INPUT_RESOLUTIONS),
        default=None,
        help="Override the model config's input resolution, and with it the "
             "decode-proxy geometry built during prep. Default: whatever the "
             "config asks for. Note that vjepa2 also needs "
             "--cfg-options model.backbone.crop=<same value>.")
    _add_resource_profile_arg(parser, include_auto=True)
    _add_common_job_args(parser)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--not-eval", action="store_true",
        help="Run the evaluation pass for its predictions only, without scoring them")
    parser.add_argument("--disable-deterministic", action="store_true",
        help="Disable deterministic for faster speed")
    parser.set_defaults(func=train)


def _add_eval_args(parser):
    _add_model_dir_arg(parser)
    _add_pair_args(parser, required=False, label="evaluation")
    _add_resource_profile_arg(parser)
    _add_common_job_args(parser, include_profile=True, include_auto_tune=True)
    parser.add_argument("--not-eval", action="store_true",
        help="Produce predictions without scoring them")
    parser.set_defaults(func=test)


def _add_predict_args(parser):
    _add_model_dir_arg(parser)
    parser.add_argument("--input", type=str, required=True,
        help="Input video file or directory of videos "
             "(supported: .mp4, .avi, .mov, .mkv, .webm)")
    parser.add_argument("--output", type=str, default=None,
        help="Directory for the prediction files (default: beside each video)")
    parser.add_argument("--include-stems", dest="include_stems", nargs="+", default=None,
        help="Restrict a directory --input to these video stems")
    parser.add_argument("--threshold", type=float, default=0.0,
        help="Minimum score for a detection to reach the prediction file")
    _add_resource_profile_arg(parser)
    _add_common_job_args(parser, include_profile=True, include_auto_tune=True)
    parser.set_defaults(func=infer)


# Separates the steps of a chain. A bare word rather than the next verb itself,
# because `--pairs` takes one-or-more values and would swallow a following
# `eval` as a filename. `then` cannot be a path or a VIDEO=CSV pair, so the
# split is unambiguous before any parsing happens.
CHAIN_SEPARATOR = "then"
# Steps that can take the model a previous step produced.
_MODEL_CONSUMERS = ("eval", "predict")


def _split_chain(argv):
    """argv -> one argv per step. Empty steps (`then then`) are an error."""
    steps, current = [], []
    for token in argv:
        if token == CHAIN_SEPARATOR:
            steps.append(current)
            current = []
        else:
            current.append(token)
    steps.append(current)
    if any(not step for step in steps):
        raise SystemExit(
            f"Empty step in the chain: `{CHAIN_SEPARATOR}` needs a command on both sides."
        )
    return steps


def _fold_eval_into_train(steps):
    """`train ... then eval ...` is ONE run, not two.

    The eval step names the videos the training loop scores each epoch against,
    and that score is what picks best.pth — so it has to be known before
    training starts. Measuring the finished model afterwards could report a
    number but could no longer choose an epoch.

    An `eval` that follows anything else, or stands alone, keeps its ordinary
    meaning: score a finished model.
    """
    folded = []
    index = 0
    while index < len(steps):
        step = steps[index]
        following = steps[index + 1] if index + 1 < len(steps) else None
        if step[0] == "train" and following and following[0] == "eval":
            merged = list(step)
            merged += ["--eval-pairs", *_parse_eval_data_step(following).pairs]
            folded.append(merged)
            index += 2
            continue
        folded.append(step)
        index += 1
    return folded


def _parse_eval_data_step(step):
    """The data flags of a chained `eval` step, as its own tiny parser.

    Deliberately narrow: inside a chain after `train`, the eval step exists to
    name videos. Its own `--model-dir` would name a model the training run is
    about to replace, so it is rejected rather than quietly ignored.
    """
    parser = argparse.ArgumentParser(prog=f"{CHAIN_SEPARATOR} eval", add_help=False)
    parser.add_argument("command")
    parser.add_argument("--pairs", dest="pairs", nargs="+", default=None)
    parsed, unknown = parser.parse_known_args(step)
    if unknown:
        raise SystemExit(
            f"`{CHAIN_SEPARATOR} eval` after `train` takes only --pairs; it names "
            f"the videos that score each epoch. Unexpected: {' '.join(unknown)}"
        )
    if not parsed.pairs:
        raise SystemExit(
            f"`{CHAIN_SEPARATOR} eval` needs --pairs naming the evaluation videos."
        )
    return parsed


def _run_chain(argv, parse):
    """Run each step in order, stopping at the first that fails.

    A step that produced a model hands its folder to any later step that did not
    name one, which is the whole reason to chain rather than paste three
    commands: the run folder is a timestamp nobody wants to copy by hand.
    """
    steps = _fold_eval_into_train(_split_chain(argv))
    model_dir = None
    for position, step in enumerate(steps, start=1):
        if step[0] in _MODEL_CONSUMERS and model_dir and "--model-dir" not in step:
            step = [*step, "--model-dir", model_dir]
        print(f"\n=== step {position}/{len(steps)}: {' '.join(step)} ===")
        args = parse(step)
        handler = getattr(args, "func", None)
        if handler is None:
            raise SystemExit(f"`{step[0]}` is not a command that can run in a chain.")
        produced = handler(args)
        if isinstance(produced, str):
            model_dir = produced
    print(f"\nChain finished: {len(steps)} step(s).")
    return model_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vtrace",
        description="V-TRACE - Video-based Temporal Recognition and Annotation "
                    "of Continuous Ethograms of Animal Behavior",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    app_parser = subparsers.add_parser("app", help="Start the TRACE annotator app")
    _add_serve_args(app_parser)

    prepare_parser = subparsers.add_parser("prepare",
        help="Download local assets needed before using TRACE")
    prepare_parser.add_argument("--weights", choices=("none", *model_weight_choices()), default="all",
        help="Model weights to download (default: all)")
    prepare_parser.set_defaults(func=prepare)

    update_parser = subparsers.add_parser("update",
        help="Check PyPI for a newer V-TRACE and optionally install it")
    update_parser.add_argument("--timeout", type=float, default=5.0,
        help="Seconds to wait for the PyPI version check (default: 5)")
    update_parser.add_argument("-y", "--yes", action="store_true",
        help="Install the new version without prompting")
    update_parser.add_argument("--check-only", action="store_true",
        help="Only report whether a newer version exists; never install")
    update_parser.set_defaults(func=update)

    train_parser = subparsers.add_parser("train", help="Train a model")
    _add_train_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    _add_eval_args(eval_parser)

    predict_parser = subparsers.add_parser("predict",
        help="Run prediction on videos (no annotations needed)")
    _add_predict_args(predict_parser)

    demo_parser = subparsers.add_parser("demo",
        help="Download and run the CalMS21 walkthrough")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command", metavar="STEP", required=True)
    demo_download = demo_subparsers.add_parser("download",
        help="Fetch the demo videos from CalMS21 and set up the checkpoint")
    demo_download.add_argument("--split", choices=["all", "train", "test"], default="all",
        help="Which videos to fetch (train ~19 GB, test ~11 GB; default: all)")
    demo_download.add_argument("--verify", action="store_true",
        help="Also CRC-check the videos already on disk, not just their size")
    demo_download.add_argument("--from", dest="source", default=None, metavar="PATH",
        help="Read a local copy of task1_videos_mp4.zip instead of downloading")
    demo_download.add_argument("--jobs", type=int, default=None, metavar="N",
        help=f"Videos to fetch at once (default: {demo_default_jobs()})")
    demo_predict = demo_subparsers.add_parser("predict",
        help="Predict on the 19 held-out demo test videos")
    demo_predict.add_argument("--model-dir", default=None, metavar="DIR",
        help="A run from `vtrace demo train` to predict with (default: the released checkpoint)")
    demo_predict.add_argument("--video", default=None, metavar="STEM",
        help="Predict on this one test video instead of all 19")
    demo_subparsers.add_parser("train",
        help="Prep and train on the demo videos, then score the test split")
    demo_parser.set_defaults(func=demo)

    argv = sys.argv[1:] if argv is None else list(argv)
    # Before the parser and before the chain is split: a `<folder>` token is the
    # same folder wherever in the line it appears, so it is looked up once.
    argv = resolve_placeholders(argv)
    if CHAIN_SEPARATOR in argv:
        return _run_chain(argv, parser.parse_args)

    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)

    if handler is None:
        # A bare `trace` at a terminal opens an interactive session; the start screen
        # names the first few commands and the prompt runs them. Anywhere else
        # (pipes, CI, `vtrace | less`) print the same screen and return, since there
        # is nobody to type at the prompt.
        from vtrace import shell, splash, version_check

        if sys.stdin.isatty() and sys.stdout.isatty():
            return shell.run(main, __version__)
        # Cache only, and no waiting: down a pipe there is nobody to act on a
        # notice, and a redirected screen is the last place to spend a network
        # round trip. A session that checked earlier leaves the answer behind.
        splash.print_start_screen(
            __version__, update=version_check.pending(__version__, wait=0)
        )
        return None
    try:
        return handler(args)
    except KeyboardInterrupt:
        # Ctrl-C on a long download or a training run is how it is meant to be
        # stopped, not a crash. The session already prints this cleanly; a
        # one-shot `vtrace ...` from a shell was dumping a socket traceback.
        print("\n  interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    main()
