"""CLI entry point for TRACE.

Usage:
    vtrace app
    vtrace prepare
    vtrace train --model maev2 --work-dir /my/data --pairs video.mp4=video.csv
    vtrace eval --model-dir /my/data/model_20260507_143012
    vtrace predict --model-dir /my/data/model_20260507_143012 --input /my/video.mp4
    vtrace pipeline --train --infer --work-dir /my/data --pairs video.mp4=video.csv --input /my/new
    vtrace update
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
from vtrace.version import __version__
from vtrace.weights import model_weight_choices


# `--model NAME` -> config file. Adding a preset is one entry here plus the config;
# the argparse choices and the pipeline spec both read this dict.
MODEL_CONFIGS = {
    "maev2": "configs/maev2.py",
    "vjepa2": "configs/vjepa2.py",
}
DEFAULT_MODEL = next(iter(MODEL_CONFIGS))
PYPI_PROJECT_NAME = "trace-tad"
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


def _prepare_pairs_into(work_dir, output_dir, args, proxy_geometry=None):
    """Prepare explicit video/CSV pairs into ``output_dir``."""
    from vtrace.data_prep import DEFAULT_PROXY_GEOMETRY, prepare_dataset

    work_dir = os.path.abspath(work_dir)
    explicit_pairs = getattr(args, "explicit_pairs", None)
    if not explicit_pairs:
        print("Error: --pairs is required. Pass each video/CSV as VIDEO_PATH=CSV_PATH.")
        sys.exit(1)

    print(f"Preparing pairs from: {work_dir}")
    print(f"Selected pairs: {', '.join(explicit_pairs)}")
    if proxy_geometry is None:
        proxy_geometry = DEFAULT_PROXY_GEOMETRY
    output_dir, dataset_json, classmap_path = prepare_dataset(
        work_dir,
        train_ratio=getattr(args, "train_ratio", 0.8),
        proxy_geometry=proxy_geometry,
        proxy_crf=getattr(args, "proxy_crf", 23),
        proxy_workers=getattr(args, "proxy_workers", None),
        explicit_pairs=explicit_pairs,
        output_dir=output_dir,
    )
    _write_prep_result(output_dir, dataset_json, classmap_path)
    print()
    return output_dir, dataset_json, classmap_path


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


def _get_access_urls(host, port):
    """Return a list of URLs where the server can be reached."""
    import socket
    urls = []
    urls.append(f"Local:   http://localhost:{port}")
    if host == "0.0.0.0" or host == "::":
        # Listening on all interfaces — discover LAN IPs
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                urls.append(f"Network: http://{ip}:{port}")
        # Fallback: connect to an external address to find the default route IP
        if len(urls) == 1:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                urls.append(f"Network: http://{ip}:{port}")
            except OSError:
                pass
    return urls


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


def _local_directories(root: Path, limit: int = _SCAN_LIMIT) -> dict:
    """basename -> [absolute paths], for directories at or under `root`."""
    index: dict[str, list[str]] = {}
    seen = 0
    stack = [(root, 0)]
    while stack and seen < limit:
        current, depth = stack.pop()
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
            index.setdefault(entry.name, []).append(entry.path)
            seen += 1
            if depth + 1 < _SCAN_DEPTH:
                stack.append((Path(entry.path), depth + 1))
    return index


def _local_context() -> str:
    """The `window.__VTRACE__` script tag injected into the served page."""
    from vtrace.demo import demo_dir

    cwd = Path.cwd()
    index = _local_directories(cwd)
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


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "")


class _GuiHandler(SimpleHTTPRequestHandler):
    """Serves the single GUI file, and nothing else on disk.

    Every path maps to that one file: the GUI routes itself with `?page=`, so
    there is nothing else to hand out, and a directory-listing server rooted at
    the user's filesystem is not something to expose even on localhost.
    """

    gui_file = None
    quiet = True
    # Filled in by `start_gui_server` when the server is loopback-only. Left None
    # for any other interface: the paths on this machine are nobody else's business.
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


DEFAULT_GUI_PORT = 8765
# Ports past the default to try before giving up. A stale server from an earlier
# session should not turn `vtrace app` into a puzzle.
_PORT_SEARCH_SPAN = 20


def _bind_first_free(host, handler):
    """Bind the first free port at or after `DEFAULT_GUI_PORT`; (None, None) if none."""
    for port in range(DEFAULT_GUI_PORT, DEFAULT_GUI_PORT + _PORT_SEARCH_SPAN):
        try:
            return ThreadingHTTPServer((host, port), handler), port
        except OSError:
            continue
    return None, None


def start_gui_server(host="127.0.0.1", port=None, verbose=False):
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
        "local_context": _local_context() if _is_loopback(host) else None,
    })
    if port is not None:
        try:
            return ThreadingHTTPServer((host, port), handler), port
        except OSError:
            return None, None
    return _bind_first_free(host, handler)


def serve_in_background(host="127.0.0.1"):
    """Start the annotator on a daemon thread; return its URL, or None.

    The interactive session brings the annotator up by itself: it is the one thing
    every user needs and there is nothing to decide about it, so making someone type
    `vtrace app` first is a step that only exists to be skipped. A daemon thread dies
    with the process, so leaving the session needs no teardown.
    """
    httpd, port = start_gui_server(host)
    if httpd is None:
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://localhost:{port}"


def serve(args):
    """Serve the V-TRACE GUI over http://localhost.

    Served rather than opened as a `file://` URL because the annotator uses the
    File System Access API and IndexedDB: `http://localhost` is a secure context
    with a stable origin, so folder permissions and the remembered last-folder
    survive a reload. A `file://` page gets neither.
    """
    if gui_path() is None:
        print("Error: the GUI file is missing from this installation "
              "(vtrace/static/gui/index.html).", file=sys.stderr)
        sys.exit(1)

    httpd, port = start_gui_server(args.host, args.port, args.verbose)
    if httpd is None:
        if args.port is not None:
            # Explicitly asked for: fail loudly rather than quietly using another port.
            print(f"\n  Cannot serve on {args.host}:{args.port} — port unavailable.\n",
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
    for line in _get_access_urls(args.host, port):
        print(f"  {line}")
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


def train(args):
    """Train a model."""
    _require_cuda()
    from vtrace.steps import TrainRequest, run_train

    model_dir = create_model_dir(args.work_dir)
    model_dir, dataset_json, classmap_path = _prepare_pairs_into(args.work_dir, model_dir, args)

    request = TrainRequest(
        config_path=_resolve_config(args),
        model_dir=model_dir,
        nproc=args.nproc,
        seed=args.seed,
        resume=args.resume,
        not_eval=args.not_eval,
        disable_deterministic=args.disable_deterministic,
        dataset_dir=model_dir,
        annotation_path=dataset_json,
        class_map=classmap_path,
        pretrained=args.pretrained,
        cfg_options=args.cfg_options,
    )

    result = run_train(request)

    if result.ok:
        print(f"\nTraining completed successfully.")
        print(f"Model directory: {model_dir}")
    else:
        print(f"\nTraining failed (exit code {result.returncode}); see {result.log_file}")
    sys.exit(result.returncode)


def test(args):
    """Evaluate a trained model."""
    _require_cuda()
    from vtrace.steps import TestRequest, run_test

    model_info = _model_info_or_exit(args.model_dir)
    dataset_dir = model_info["model_dir"]
    annotation_path = model_info["dataset_json"]
    if args.explicit_pairs and not args.work_dir:
        print("Error: --pairs requires --work-dir for evaluation data.")
        sys.exit(1)
    output_dir = create_eval_dir(model_info["model_dir"])
    if args.work_dir:
        dataset_dir, annotation_path, _ = _prepare_pairs_into(
            args.work_dir,
            output_dir,
            args,
        )
    elif not annotation_path:
        print("Error: model_dir has no dataset.json. Pass --work-dir and --pairs for evaluation data.")
        sys.exit(1)

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
        cfg_options=args.cfg_options,
    )

    result = run_test(request)

    if result.ok:
        print(f"\nTesting completed successfully.")
    else:
        print(f"\nTesting failed (exit code {result.returncode}); see {result.log_file}")
    sys.exit(result.returncode)


def infer(args):
    """Run inference on video files."""
    _require_cuda()
    from vtrace.steps import InferRequest, run_infer

    model_info = _model_info_or_exit(args.model_dir)

    request = InferRequest(
        model_dir=model_info["model_dir"],
        input=args.input,
        output=args.output,
        seed=args.seed,
        profile=args.profile,
        auto_tune=args.auto_tune,
        threshold=args.threshold,
        cfg_options=args.cfg_options,
    )

    result = run_infer(request)

    if result.ok:
        print(f"\nInference completed successfully.")
    else:
        print(f"\nInference failed (exit code {result.returncode}); see {result.log_file}")
    sys.exit(result.returncode)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _or_exit(result, label):
    """Stop the pipeline at the first failing step."""
    if not result.ok:
        print(f"\n{label} failed (exit code {result.returncode}); see {result.log_file}")
        sys.exit(result.returncode)
    return result


def _run_pipeline_spec(spec):
    """Run the UI-shaped pipeline spec step by step."""
    from vtrace.steps import (
        InferRequest,
        PrepRequest,
        TestRequest,
        TrainRequest,
        TrainTuneRequest,
        run_infer,
        run_prep,
        run_test,
        run_train,
        run_train_tune,
    )
    from vtrace.model_artifacts import resolve_model_dir
    from vtrace.pipeline_plan import (
        eval_resource_cfg_options,
        eval_resource_settings,
        prep_pairs,
        prep_work_dir,
        resource_profile_by_name,
        resource_settings_from_profile,
        train_resource_cfg_options,
        train_resource_settings,
    )

    if spec.steps.train or spec.steps.extra_test or spec.steps.infer:
        _require_cuda()

    prep_result = None
    active_model = None
    config_path = _resolve_config(
        argparse.Namespace(model=spec.model or DEFAULT_MODEL, config=spec.config or None)
    )

    # Derive the proxy geometry from the *merged* config rather than from the
    # spec's raw int, so a `--cfg-options dataset.*.pipeline.*.resize=` override
    # can't leave prep building proxies at a resolution training won't use.
    from vtrace.config import Config
    from vtrace.proxy_geometry import config_geometry, input_resize_cfg_options

    resolved_cfg = Config.fromfile(config_path)
    if spec.input_resolution:
        resolved_cfg.merge_from_dict(
            input_resize_cfg_options(resolved_cfg, spec.input_resolution)
        )
    proxy_geometry = config_geometry(resolved_cfg)

    if spec.steps.train or spec.steps.extra_test:
        print("\n--- Preparing dataset ---")
        prep_proxy_workers = (
            eval_resource_settings(spec).num_workers
            if spec.steps.extra_test
            else train_resource_settings(spec).num_workers
        )
        prep_result_step = _or_exit(run_prep(PrepRequest(
            work_dir=prep_work_dir(spec),
            train_ratio=spec.train_ratio,
            proxy_resolution=proxy_geometry.short_side,
            proxy_aspect=not proxy_geometry.square,
            proxy_workers=prep_proxy_workers,
            explicit_pairs=prep_pairs(spec),
        )), "Dataset prep")
        prep_result = _read_json(os.path.join(prep_result_step.work_dir, "prep_result.json"))

    if spec.steps.train:
        print("\n--- Training ---")
        cfg_options = {
            "scheduler.max_epoch": spec.epochs,
            "workflow.end_epoch": spec.epochs,
            "workflow.val_start_epoch": spec.val_start_epoch,
            "workflow.val_eval_interval": spec.val_interval,
        }

        if spec.resource_profile == "auto":
            print("\n--- Tuning train resources ---")
            tune_step = _or_exit(run_train_tune(TrainTuneRequest(
                config_path=config_path,
                model_dir=prep_result["model_dir"],
                annotation_path=prep_result["dataset_json"],
                class_map=prep_result["classmap_path"],
            )), "Train resource tuning")
            tune_result = _read_json(os.path.join(tune_step.work_dir, "train_tune_result.json"))
            resource_profile = resource_profile_by_name(tune_result.get("recommended_profile"))
            train_settings = resource_settings_from_profile(resource_profile.id)
        else:
            train_settings = train_resource_settings(spec)

        cfg_options.update(train_resource_cfg_options(
            train_settings,
            spec.input_resolution,
            resolved_cfg,
        ))

        _or_exit(run_train(TrainRequest(
            config_path=config_path,
            model_dir=prep_result["model_dir"],
            dataset_dir=prep_result["model_dir"],
            annotation_path=prep_result["dataset_json"],
            class_map=prep_result["classmap_path"],
            cfg_options=cfg_options,
        )), "Training")
        active_model = resolve_model_dir(prep_result["model_dir"])

    if not spec.steps.train and (spec.steps.extra_test or spec.steps.infer):
        active_model = resolve_model_dir(spec.model_dir)

    if spec.steps.extra_test:
        print("\n--- Extra test ---")
        test_request = TestRequest(
            model_dir=active_model["model_dir"],
            auto_tune=False,
            cfg_options=eval_resource_cfg_options(eval_resource_settings(spec), resolved_cfg),
        )
        if prep_result:
            test_request.dataset_dir = prep_result["model_dir"]
            test_request.annotation_path = prep_result["dataset_json"]
        _or_exit(run_test(test_request), "Extra test")

    if spec.steps.infer:
        print("\n--- Inference ---")
        _or_exit(run_infer(InferRequest(
            model_dir=active_model["model_dir"],
            input=spec.input_selection.folder,
            included_stems=spec.input_selection.stems,
            threshold=spec.threshold,
            auto_tune=False,
            cfg_options=eval_resource_cfg_options(eval_resource_settings(spec), resolved_cfg),
        )), "Inference")

    print("\nPipeline completed successfully.")


def run(args):
    """Run the pipeline: prep -> train -> test -> predict."""
    from vtrace.pipeline_plan import PipelineSpecError, spec_from_cli_args, validate_pipeline_spec

    spec = spec_from_cli_args(args)
    try:
        validate_pipeline_spec(spec)
    except PipelineSpecError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    _run_pipeline_spec(spec)


def demo(args):
    """Dispatch `vtrace demo <step>`."""
    from vtrace import demo as demo_mod

    step = {"download": demo_mod.download, "predict": demo_mod.predict, "train": demo_mod.train}
    return step[args.demo_command](args)


def _add_serve_args(parser):
    parser.add_argument("--host", default="127.0.0.1",
        help="Interface to serve on (default: 127.0.0.1, this computer only)")
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


def _add_work_dir_arg(parser, *, required=True):
    parser.add_argument("--work-dir", type=str, required=required,
        help="Directory containing video/CSV files. Relative --pairs are resolved against this path.")


def _add_model_dir_arg(parser):
    parser.add_argument("--model-dir", type=str, required=True,
        help="Model artifact directory produced by `vtrace train`")


def _add_pair_args(parser, *, required=True):
    parser.add_argument("--pairs", dest="explicit_pairs",
        nargs="+", required=required, metavar="VIDEO=CSV",
        help="Explicit video/annotation pairs to use from --work-dir. Each item "
             "must be VIDEO_PATH=CSV_PATH. Relative paths are resolved against "
             "--work-dir; absolute paths are accepted.")


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
    _add_work_dir_arg(parser)
    _add_pair_args(parser)
    parser.add_argument("--pretrained", type=str, default=None,
        help="Pretrained backbone weights path (overrides config's pretrain)")
    _add_common_job_args(parser, include_nproc=True)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--not-eval", action="store_true", help="Skip evaluation, inference only")
    parser.add_argument("--disable-deterministic", action="store_true",
        help="Disable deterministic for faster speed")
    parser.set_defaults(func=train)


def _add_eval_args(parser):
    _add_model_dir_arg(parser)
    _add_work_dir_arg(parser, required=False)
    _add_pair_args(parser, required=False)
    parser.add_argument("--cache-workers", type=int, default=None,
        help="Parallel workers for cached evaluation clip writing")
    _add_common_job_args(parser, include_nproc=True, include_profile=True, include_auto_tune=True)
    parser.add_argument("--not-eval", action="store_true", help="Skip evaluation, inference only")
    parser.set_defaults(func=test)


def _add_predict_args(parser):
    _add_model_dir_arg(parser)
    parser.add_argument("--input", type=str, required=True,
        help="Input video file or directory of videos "
             "(supported: .mp4, .avi, .mov, .mkv, .webm)")
    parser.add_argument("--output", type=str, default=None,
        help="Directory for the prediction files (default: beside each video)")
    parser.add_argument("--threshold", type=float, default=0.0,
        help="Minimum score for a detection to reach the prediction file")
    _add_common_job_args(parser, include_profile=True, include_auto_tune=True)
    parser.set_defaults(func=infer)


def _add_pipeline_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="Override config settings")
    _add_model_config_args(parser)
    parser.add_argument("--train", action="store_true",
        help="Train a model")
    parser.add_argument("--extra-test", action="store_true",
        help="Run an additional evaluation pass")
    parser.add_argument("--infer", action="store_true",
        help="Run inference")
    parser.add_argument("--model-dir", type=str, default=None,
        help="Model artifact directory when not training")
    parser.add_argument("--work-dir", type=str, default=None,
        help="Dataset folder for train or extra-test prep")
    parser.add_argument("--pairs", dest="explicit_pairs", nargs="+", metavar="VIDEO=CSV",
        help="Explicit video/annotation pairs for train or extra-test prep")
    parser.add_argument("--input-resolution", type=int, choices=[112, 144, 160, 192, 224, 256], default=None,
        help="Override the model config's input resolution, and with it the "
             "decode-proxy geometry built during prep. Default: whatever the "
             "config asks for. Note that vjepa2 also needs "
             "--cfg-options model.backbone.crop=<same value>.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
        help="Train/validation split ratio for prep (default: 0.8)")
    parser.add_argument("--epochs", type=int, default=100,
        help="Total training epochs (default: 100)")
    parser.add_argument("--val-start-epoch", type=int, default=50,
        help="Validation start epoch (default: 50)")
    parser.add_argument("--val-interval", type=int, default=10,
        help="Validation interval in epochs (default: 10)")
    parser.add_argument("--resource-profile", choices=["auto", "low", "balanced", "high"], default="balanced",
        help="Dataloader resource profile for every step (default: balanced). "
             "`auto` benchmarks the training dataloader first; evaluation stays balanced. "
             "Use --cfg-options for finer control.")
    parser.add_argument("--input", type=str, default=None,
        help="Input video file or folder for inference")
    parser.add_argument("--include-stems", dest="include_stems", nargs="+",
        help="Restrict inference to selected video stems")
    parser.add_argument("--threshold", type=float, default=0.0,
        help="Minimum prediction score for pipeline inference outputs")
    parser.set_defaults(func=run)


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

    pipeline_parser = subparsers.add_parser("pipeline",
        help="Run the pipeline: prep -> train -> test -> predict")
    _add_pipeline_args(pipeline_parser)

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
