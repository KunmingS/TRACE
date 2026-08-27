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


# ── Local path context handed to the page ───────────────────────────────────
# A browser never learns where a folder the user picked actually lives: the File
# System Access API hands the page a capability (a handle it can read) and not a
# location, deliberately, because absolute paths leak the user's name and layout.
# But this server IS the user's machine, so it can name the directories it can see
# and let the page match a picked folder against them by name. That turns the
# generated command from a template with `/path/to/...` into one that runs as
# pasted, without adding an API the page could ask arbitrary questions through.
_SCAN_DEPTH = 3
_SCAN_LIMIT = 4000
_SCAN_SKIP = {".git", "__pycache__", "node_modules", "venv", ".venv", "site-packages"}


def _scan_roots() -> list[Path]:
    """Where to look for the folder the user picks in the page.

    A data folder beside the checkout is as common as one inside it, and neither is
    reachable from the other by walking down, so the parent and home are searched
    too. Ordered by relevance: the entry budget is spent in order.
    """
    cwd = Path.cwd()
    roots: list[Path] = []
    for candidate in (cwd, cwd.parent, Path.home()):
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
            if entry.name.startswith(".") or entry.name in _SCAN_SKIP:
                continue
            paths = index.setdefault(entry.name, [])
            if entry.path not in paths:
                paths.append(entry.path)
            seen += 1
            if depth + 1 < _SCAN_DEPTH:
                queue.append((Path(entry.path), depth + 1))
    return index


def _local_context() -> str:
    """The `window.__VTRACE__` script tag injected into the served page."""
    from vtrace.demo import demo_dir

    cwd = Path.cwd()
    index = _local_directories(_scan_roots())
    demo = demo_dir()
    if demo.is_dir():
        for extra in (demo, demo / "videos" / "train", demo / "videos" / "test"):
            if extra.is_dir():
                index.setdefault(extra.name, [])
                if str(extra) not in index[extra.name]:
                    index[extra.name].append(str(extra))
    payload = {"cwd": str(cwd), "dirs": index}
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
    # Filled in by `start_gui_server`. The server is loopback-only, so these
    # paths never leave the machine they describe.
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
    Two things it does not get. It has no `window.__VTRACE__`, the local
    directory index that turns a folder picked through a capability-only API
    back into the absolute path the copyable command needs. And every `file://`
    page on the machine shares one origin, so a second copy of this file — or
    any other local page — reads the same IndexedDB, down to the stored
    directory handles. `http://localhost:PORT` supplies the index and an origin
    of its own.

    Loopback only. Serving to a LAN would hand out neither: the directory index
    names paths on the server's disk, which is not where a remote viewer's
    videos are, so a remote page is the bare file with extra steps.
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

    model_dir = create_model_dir(args.output)
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
        nproc=args.nproc,
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
        nproc=args.nproc,
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


def _add_common_job_args(parser, *, include_nproc=False, include_profile=False, include_auto_tune=False):
    if include_nproc:
        parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs (default: 1)")
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
    # Required, not derived from where the videos happen to live. A corpus is an
    # input that many runs read; which of them writes its checkpoints beside it
    # is a decision, and an unasked one puts run folders wherever the data sits.
    parser.add_argument("--output", type=str, required=True,
        help="Directory to create the run folder in. The run writes "
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
    _add_common_job_args(parser, include_nproc=True)
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
    _add_common_job_args(parser, include_nproc=True, include_profile=True, include_auto_tune=True)
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
    demo_subparsers.add_parser("predict", help="Predict on a held-out demo video")
    demo_subparsers.add_parser("train", help="Prep and train on the demo videos")
    demo_parser.set_defaults(func=demo)

    argv = sys.argv[1:] if argv is None else list(argv)
    if CHAIN_SEPARATOR in argv:
        return _run_chain(argv, parser.parse_args)

    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)

    if handler is None:
        # A bare `trace` at a terminal opens an interactive session; the start screen
        # names the first few commands and the prompt runs them. Anywhere else
        # (pipes, CI, `vtrace | less`) print the same screen and return, since there
        # is nobody to type at the prompt.
        from vtrace import shell, splash

        if sys.stdin.isatty() and sys.stdout.isatty():
            return shell.run(main, __version__)
        splash.print_start_screen(__version__)
        return None
    return handler(args)


if __name__ == "__main__":
    main()
