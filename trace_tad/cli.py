"""CLI entry point for TRACE.

Usage:
    trace app
    trace prepare
    trace train --model large --work-dir /my/data --pairs video.mp4=video.csv
    trace eval --model-dir /my/data/model_20260507_143012
    trace predict --model-dir /my/data/model_20260507_143012 --input /my/video.mp4
    trace pipeline <config>
    trace update
"""
import argparse
import json
import os
import shutil
import subprocess
import signal
import sys
import time

from trace_tad.config import DictAction
from trace_tad.model_artifacts import (
    create_eval_dir,
    create_model_dir,
    resolve_model_dir,
)
from trace_tad.version import __version__
from trace_tad.weights import model_weight_choices


MODEL_CONFIGS = {
    "small": "configs/small.py",
    "large": "configs/large.py",
}
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
    model = getattr(args, "model", "small")
    rel = MODEL_CONFIGS[model]
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


def _prepare_pairs_into(work_dir, output_dir, args, force_cache_mode=None):
    """Prepare explicit video/CSV pairs into ``output_dir``."""
    from trace_tad.data_prep import prepare_dataset

    work_dir = os.path.abspath(work_dir)
    explicit_pairs = getattr(args, "explicit_pairs", None)
    if not explicit_pairs:
        print("Error: --pairs is required. Pass each video/CSV as VIDEO_PATH=CSV_PATH.")
        sys.exit(1)

    print(f"Preparing pairs from: {work_dir}")
    print(f"Selected pairs: {', '.join(explicit_pairs)}")
    reencode_clips = getattr(args, "reencode_clips", False)
    cache_mode = force_cache_mode or getattr(args, "cache_mode", None)
    if cache_mode is None:
        cache_mode = "physical" if reencode_clips else "virtual"
    output_dir, dataset_json, classmap_path = prepare_dataset(
        work_dir,
        clip_frames=getattr(args, "clip_frames", 768),
        train_ratio=getattr(args, "train_ratio", 0.8),
        virtual_clips=cache_mode == "virtual",
        cache_mode=cache_mode,
        cache_resolution=getattr(args, "cache_resolution", 144),
        cache_crf=getattr(args, "cache_crf", 23),
        cache_workers=getattr(args, "cache_workers", None),
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


def find_annotator_dir():
    """Locate the trace-annotator directory."""
    # Check relative to the package install location
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotator_dir = os.path.join(pkg_dir, "trace-annotator")
    if os.path.isdir(annotator_dir):
        return annotator_dir

    # Check relative to cwd
    cwd_annotator = os.path.join(os.getcwd(), "trace-annotator")
    if os.path.isdir(cwd_annotator):
        return cwd_annotator

    return None


def _find_bundled_index():
    """Check if pre-built frontend assets exist in the package."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "annotator")
    index = os.path.join(static_dir, "index.html")
    if os.path.isfile(index):
        return static_dir
    return None


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


def serve(args):
    """Start the TRACE annotation server."""
    annotator_dir = find_annotator_dir()
    processes = []

    if args.port is None:
        args.port = 3001 if args.dev else 8000

    def cleanup(signum=None, frame=None):
        for p in processes:
            try:
                p.terminate()
            except OSError:
                pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if args.dev:
        # ── Dev mode: two processes (uvicorn + Vite) ──
        if annotator_dir is None:
            print("Error: Cannot find trace-annotator directory (required for --dev mode).")
            sys.exit(1)

        try:
            # Start backend
            if not args.frontend_only:
                print(f"Starting backend server on port {args.port}...")
                backend_proc = subprocess.Popen(
                    [
                        sys.executable, "-m", "uvicorn",
                        "trace_tad.server.app:app",
                        "--host", args.host,
                        "--port", str(args.port),
                        "--reload",
                    ],
                )
                processes.append(backend_proc)

            # Start frontend
            if not args.backend_only:
                node_modules = os.path.join(annotator_dir, "node_modules")
                if not os.path.isdir(node_modules):
                    print("Installing frontend dependencies (first run)...")
                    subprocess.check_call(["npm", "install"], cwd=annotator_dir)

                print(f"Starting frontend dev server on port {args.frontend_port}...")
                env = os.environ.copy()
                env["PORT"] = str(args.frontend_port)
                env["VITE_PROXY_API_TARGET"] = f"http://localhost:{args.port}"
                frontend_proc = subprocess.Popen(
                    ["npm", "run", "dev"],
                    cwd=annotator_dir,
                    env=env,
                )
                processes.append(frontend_proc)

            if processes:
                print()
                if not args.frontend_only:
                    print(f"  Backend:  http://{args.host}:{args.port}")
                if not args.backend_only:
                    print(f"  Frontend: http://localhost:{args.frontend_port}")
                print()
                for url in _get_access_urls(args.host, args.port):
                    print(f"  {url}")
                print()
                print("Press Ctrl+C to stop.")
                print()

                while True:
                    for p in processes:
                        ret = p.poll()
                        if ret is not None:
                            print(f"Process {p.args} exited with code {ret}")
                            cleanup()
                    time.sleep(1)

        except KeyboardInterrupt:
            cleanup()

    else:
        # ── Production mode: single uvicorn process ──
        bundled = _find_bundled_index()
        if bundled is None:
            print("Warning: No bundled frontend found.")
            print("  The API will start, but there is no frontend to serve.")
            print("  Run 'trace dev build-frontend' first, or use 'trace app --dev'.")
            print()

        print(f"Starting TRACE server on http://{args.host}:{args.port}")
        if bundled:
            print(f"  Serving frontend from: {bundled}")
        print()
        for url in _get_access_urls(args.host, args.port):
            print(f"  {url}")
        print()
        print("Press Ctrl+C to stop.")
        print()

        try:
            backend_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "trace_tad.server.app:app",
                    "--host", args.host,
                    "--port", str(args.port),
                ],
            )
            processes.append(backend_proc)
            backend_proc.wait()
        except KeyboardInterrupt:
            cleanup()


def build_frontend(args):
    """Build the frontend and copy into the package for distribution."""
    annotator_dir = find_annotator_dir()
    if annotator_dir is None:
        print("Error: Cannot find trace-annotator directory.")
        sys.exit(1)

    dist_dir = os.path.join(annotator_dir, "dist")
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "annotator")

    # Build unless --skip-npm
    if not args.skip_npm:
        node_modules = os.path.join(annotator_dir, "node_modules")
        if not os.path.isdir(node_modules):
            print("Installing frontend dependencies...")
            subprocess.check_call(["npm", "install"], cwd=annotator_dir)

        print("Building frontend...")
        subprocess.check_call(["npm", "run", "build"], cwd=annotator_dir)

    if not os.path.isdir(dist_dir):
        print(f"Error: Build output not found at {dist_dir}")
        sys.exit(1)

    # Copy dist → trace_tad/static/annotator/
    if os.path.isdir(static_dir):
        shutil.rmtree(static_dir)
    shutil.copytree(dist_dir, static_dir)

    print(f"Frontend assets copied to: {static_dir}")
    print("Done. Run 'trace app' to start the server.")


def _download_weights_selection(selection):
    """Download the selected model weights and print their local paths."""
    from trace_tad.weights import download_model_weights

    paths = download_model_weights(selection)
    for path in paths:
        print(path)
    return paths


def prepare(args):
    """Prepare local assets needed by TRACE."""
    if args.weights == "none":
        print("No prepare tasks selected.")
        return []

    print(f"Preparing model weights: {args.weights}")
    paths = _download_weights_selection(args.weights)
    print()
    print("TRACE is ready.")
    return paths


def update(args):
    """Check whether the installed TRACE package matches the PyPI version."""
    try:
        latest = _fetch_latest_pypi_version(timeout=args.timeout)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    current = __version__
    if latest == current:
        print(f"TRACE is up to date ({current}).")
    else:
        print(f"TRACE {latest} is available on PyPI.")
        print(f"Installed version: {current}")
        print()
        print("Update with:")
        print(f"  python -m pip install --upgrade {PYPI_PROJECT_NAME}")
    return 0


def train(args):
    """Run a training job."""
    _require_cuda()
    from trace_tad.jobs import JobManager, TrainRequest

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

    manager = JobManager(max_concurrency=1)
    job = manager.start_train_job(request)
    print(f"Job {job.job_id} submitted (log: {job.log_file})")
    print()

    job = manager.wait_for_job(job.job_id, stream_to=sys.stdout)

    if job.status.value == "completed":
        print(f"\nTraining completed successfully.")
        print(f"Model directory: {model_dir}")
    else:
        print(f"\nTraining {job.status.value}: {job.error_message or ''}")
    sys.exit(job.return_code or (0 if job.status.value == "completed" else 1))


def test(args):
    """Run a test/inference job."""
    _require_cuda()
    from trace_tad.jobs import JobManager, TestRequest

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
            force_cache_mode="cached_video",
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

    manager = JobManager(max_concurrency=1)
    job = manager.start_test_job(request)
    print(f"Job {job.job_id} submitted (log: {job.log_file})")
    print()

    job = manager.wait_for_job(job.job_id, stream_to=sys.stdout)

    if job.status.value == "completed":
        print(f"\nTesting completed successfully.")
    else:
        print(f"\nTesting {job.status.value}: {job.error_message or ''}")
    sys.exit(job.return_code or (0 if job.status.value == "completed" else 1))


def infer(args):
    """Run inference on video files."""
    _require_cuda()
    from trace_tad.jobs import JobManager, InferRequest

    model_info = _model_info_or_exit(args.model_dir)

    request = InferRequest(
        model_dir=model_info["model_dir"],
        input=args.input,
        output=args.output,
        seed=args.seed,
        profile=args.profile,
        auto_tune=args.auto_tune,
        annotated_video=args.annotated_video,
        threshold=args.threshold,
        cfg_options=args.cfg_options,
    )

    manager = JobManager(max_concurrency=1)
    job = manager.start_infer_job(request)
    print(f"Job {job.job_id} submitted (log: {job.log_file})")
    print()

    job = manager.wait_for_job(job.job_id, stream_to=sys.stdout)

    if job.status.value == "completed":
        print(f"\nInference completed successfully.")
    else:
        print(f"\nInference {job.status.value}: {job.error_message or ''}")
    sys.exit(job.return_code or (0 if job.status.value == "completed" else 1))


def _pipeline_spec_mode_requested(args):
    """Return True when `trace pipeline` is using the UI-shaped flag mode."""
    return (
        not getattr(args, "config", None)
        or getattr(args, "train", False)
        or getattr(args, "extra_test", False)
        or getattr(args, "infer", False)
        or bool(getattr(args, "work_dir", None))
        or bool(getattr(args, "explicit_pairs", None))
        or bool(getattr(args, "model_dir", None))
        or bool(getattr(args, "input", None))
        or bool(getattr(args, "include_stems", None))
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _wait_or_exit(manager, job, label):
    print(f"Job {job.job_id} submitted (log: {job.log_file})")
    print()
    completed = manager.wait_for_job(job.job_id, stream_to=sys.stdout)
    if completed.status.value != "completed":
        print(f"\n{label} {completed.status.value}: {completed.error_message or ''}")
        sys.exit(completed.return_code or 1)
    return completed


def _run_pipeline_spec(spec):
    """Run the UI-shaped pipeline spec through the same job queue primitives."""
    from trace_tad.jobs import (
        JobManager,
        PrepRequest,
        TrainRequest,
        TrainTuneRequest,
        TestRequest,
        InferRequest,
    )
    from trace_tad.model_artifacts import resolve_model_dir
    from trace_tad.pipeline_plan import (
        prep_pairs,
        prep_work_dir,
        eval_resource_cfg_options,
        prep_cache_mode,
        resource_profile_by_name,
        resource_settings_from_profile,
        train_resource_cfg_options,
        train_resource_settings,
    )

    if spec.steps.train or spec.steps.extra_test or spec.steps.infer:
        _require_cuda()

    manager = JobManager(max_concurrency=1)
    prep_result = None
    active_model = None
    config_path = _resolve_config(argparse.Namespace(model=spec.model_size, config=None))

    if spec.steps.train or spec.steps.extra_test:
        print("\n--- Preparing dataset ---")
        prep_cache_workers = (
            spec.resources.test.num_workers
            if spec.steps.extra_test
            else train_resource_settings(spec).num_workers
        )
        prep_job = manager.start_prep_job(PrepRequest(
            work_dir=prep_work_dir(spec),
            train_ratio=spec.train_ratio,
            cache_mode=prep_cache_mode(spec),
            cache_resolution=spec.cache_resolution,
            cache_workers=prep_cache_workers,
            explicit_pairs=prep_pairs(spec),
        ))
        completed_prep = _wait_or_exit(manager, prep_job, "Dataset prep")
        prep_result = _read_json(os.path.join(completed_prep.work_dir, "prep_result.json"))

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
            tune_job = manager.start_train_tune_job(TrainTuneRequest(
                config_path=config_path,
                model_dir=prep_result["model_dir"],
                annotation_path=prep_result["dataset_json"],
                class_map=prep_result["classmap_path"],
            ))
            completed_tune = _wait_or_exit(manager, tune_job, "Train resource tuning")
            tune_result = _read_json(os.path.join(completed_tune.work_dir, "train_tune_result.json"))
            resource_profile = resource_profile_by_name(tune_result.get("recommended_profile"))
            train_settings = resource_settings_from_profile(resource_profile.id)
        else:
            train_settings = train_resource_settings(spec)

        cfg_options.update(train_resource_cfg_options(
            train_settings,
            spec.cache_resolution,
            spec.model_size,
        ))

        train_job = manager.start_train_job(TrainRequest(
            config_path=config_path,
            model_dir=prep_result["model_dir"],
            dataset_dir=prep_result["model_dir"],
            annotation_path=prep_result["dataset_json"],
            class_map=prep_result["classmap_path"],
            cfg_options=cfg_options,
        ))
        _wait_or_exit(manager, train_job, "Training")
        active_model = resolve_model_dir(prep_result["model_dir"])

    if not spec.steps.train and (spec.steps.extra_test or spec.steps.infer):
        active_model = resolve_model_dir(spec.model_dir)

    if spec.steps.extra_test:
        print("\n--- Extra test ---")
        test_request = TestRequest(
            model_dir=active_model["model_dir"],
            auto_tune=False,
            cfg_options=eval_resource_cfg_options(spec.resources.test),
        )
        if prep_result:
            test_request.dataset_dir = prep_result["model_dir"]
            test_request.annotation_path = prep_result["dataset_json"]
        test_job = manager.start_test_job(test_request)
        _wait_or_exit(manager, test_job, "Extra test")

    if spec.steps.infer:
        print("\n--- Inference ---")
        infer_job = manager.start_infer_job(InferRequest(
            model_dir=active_model["model_dir"],
            input=spec.input_selection.folder,
            included_stems=spec.input_selection.stems,
            annotated_video=spec.annotated_video,
            threshold=spec.threshold,
            auto_tune=False,
            cfg_options=eval_resource_cfg_options(spec.resources.infer),
        ))
        _wait_or_exit(manager, infer_job, "Inference")

    print("\nPipeline completed successfully.")


def run(args):
    """Run the TRACE pipeline: train -> infer -> export."""
    if _pipeline_spec_mode_requested(args):
        from trace_tad.pipeline_plan import PipelineSpecError, spec_from_cli_args, validate_pipeline_spec

        spec = spec_from_cli_args(args)
        try:
            validate_pipeline_spec(spec)
        except PipelineSpecError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        return _run_pipeline_spec(spec)

    # Determine mode
    _require_cuda()
    from trace_tad.pipeline import run_pipeline

    if args.train_only:
        mode = "train_only"
    elif args.infer_only:
        mode = "infer_only"
    else:
        mode = "full"

    if mode == "infer_only" and not args.checkpoint:
        print("Error: --checkpoint is required with --infer-only")
        sys.exit(1)

    state = run_pipeline(
        config=args.config,
        mode=mode,
        checkpoint=args.checkpoint,
        infer_videos=args.infer_videos or [],
        export_format=args.export,
        cfg_options=args.cfg_options or {},
        seed=args.seed,
    )

    if state.phase == "failed":
        print(f"\nPipeline failed: {state.error}")
        sys.exit(1)


def _add_serve_args(parser, *, default_dev=False):
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
        help="Server port (default: 8000 in prod, 3001 in dev)")
    if default_dev:
        parser.set_defaults(dev=True)
    else:
        parser.add_argument("--dev", action="store_true",
            help="Development mode: start Vite dev server + backend separately (requires Node.js)")
    parser.add_argument("--frontend-port", type=int, default=3000,
        help="Frontend port in dev mode (default: 3000)")
    parser.add_argument("--backend-only", action="store_true",
        help="Only start backend (dev mode only)")
    parser.add_argument("--frontend-only", action="store_true",
        help="Only start frontend (dev mode only)")
    parser.set_defaults(func=serve)


def _add_build_frontend_args(parser):
    parser.add_argument("--skip-npm", action="store_true",
        help="Skip npm build, just copy existing dist/")
    parser.set_defaults(func=build_frontend)


def _add_model_config_args(parser):
    parser.add_argument("--model", type=str, choices=["small", "large"], default="small",
        help="Model size (default: small)")
    parser.add_argument("--config", type=str, default=None,
        help="Custom config file path (overrides --model)")


def _add_work_dir_arg(parser, *, required=True):
    parser.add_argument("--work-dir", type=str, required=required,
        help="Directory containing video/CSV files. Relative --pairs are resolved against this path.")


def _add_model_dir_arg(parser):
    parser.add_argument("--model-dir", type=str, required=True,
        help="Model artifact directory produced by `trace train`")


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
    parser.add_argument("--reencode-clips", action="store_true",
        help="During dataset prep, physically extract each clip with ffmpeg "
             "(CRF-18 re-encode). Default is virtual clips: zero quality loss, "
             "~10x faster prep, half the disk usage. Use this flag if you need "
             "self-contained clip files (e.g. shipping the dataset elsewhere).")
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
        help="Output JSON path (default: predictions.json in work_dir)")
    parser.add_argument("--annotated-video", action="store_true",
        help="Render annotated MP4 video(s) with prediction overlays")
    parser.add_argument("--threshold", type=float, default=0.0,
        help="Minimum prediction score for JSON, CSV, and annotated video output")
    _add_common_job_args(parser, include_profile=True, include_auto_tune=True)
    parser.set_defaults(func=infer)


def _add_pipeline_args(parser):
    parser.add_argument("config", metavar="FILE", type=str, nargs="?",
        help="Path to config file (legacy pipeline mode)")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--infer-only", action="store_true", help="Only run inference (requires --checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for inference")
    parser.add_argument("--infer-videos", nargs="+", default=None, help="Video file paths for inference")
    parser.add_argument("--export", choices=["csv"], default=None, help="Export results format")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="Override config settings")
    parser.add_argument("--model", type=str, choices=["small", "large"], default="small",
        help="Model size for UI-shaped pipeline mode (default: small)")
    parser.add_argument("--train", action="store_true",
        help="UI-shaped pipeline mode: train a model")
    parser.add_argument("--extra-test", action="store_true",
        help="UI-shaped pipeline mode: run an additional evaluation pass")
    parser.add_argument("--infer", action="store_true",
        help="UI-shaped pipeline mode: run inference")
    parser.add_argument("--model-dir", type=str, default=None,
        help="Model artifact directory when not training")
    parser.add_argument("--work-dir", type=str, default=None,
        help="Dataset folder for train or extra-test prep")
    parser.add_argument("--pairs", dest="explicit_pairs", nargs="+", metavar="VIDEO=CSV",
        help="Explicit video/annotation pairs for train or extra-test prep")
    parser.add_argument("--cache-mode", choices=["cached_video", "virtual"], default="cached_video",
        help="Dataset cache mode for prep (default: cached_video)")
    parser.add_argument("--cache-resolution", type=int, choices=[112, 144, 192, 224], default=144,
        help="Square resolution for cached_video clips (default: 144)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
        help="Train/validation split ratio for prep (default: 0.8)")
    parser.add_argument("--epochs", type=int, default=100,
        help="Total training epochs (default: 100)")
    parser.add_argument("--val-start-epoch", type=int, default=50,
        help="Validation start epoch (default: 50)")
    parser.add_argument("--val-interval", type=int, default=10,
        help="Validation interval in epochs (default: 10)")
    parser.add_argument("--resource-profile", choices=["auto", "low", "balanced", "high"], default="balanced",
        help="Training dataloader resource profile (default: balanced)")
    parser.add_argument("--train-workers", type=int, default=None,
        help="Override training dataloader workers")
    parser.add_argument("--train-decode-threads", type=int, default=None,
        help="Override training video decode threads")
    parser.add_argument("--train-prefetch", type=int, default=None,
        help="Override training dataloader prefetch factor")
    parser.add_argument("--test-resource-profile", choices=["low", "balanced", "high"], default="balanced",
        help="Evaluation dataloader resource profile (default: balanced)")
    parser.add_argument("--test-batch-size", type=int, default=None,
        help="Override evaluation batch size")
    parser.add_argument("--test-workers", type=int, default=None,
        help="Override evaluation dataloader workers")
    parser.add_argument("--test-decode-threads", type=int, default=None,
        help="Override evaluation video decode threads")
    parser.add_argument("--test-prefetch", type=int, default=None,
        help="Override evaluation dataloader prefetch factor")
    parser.add_argument("--infer-resource-profile", choices=["low", "balanced", "high"], default="balanced",
        help="Prediction dataloader resource profile (default: balanced)")
    parser.add_argument("--infer-batch-size", type=int, default=None,
        help="Override prediction batch size")
    parser.add_argument("--infer-workers", type=int, default=None,
        help="Override prediction dataloader workers")
    parser.add_argument("--infer-decode-threads", type=int, default=None,
        help="Override prediction video decode threads")
    parser.add_argument("--infer-prefetch", type=int, default=None,
        help="Override prediction dataloader prefetch factor")
    parser.add_argument("--input", type=str, default=None,
        help="Input video file or folder for inference")
    parser.add_argument("--include-stems", dest="include_stems", nargs="+",
        help="Restrict inference to selected video stems")
    parser.add_argument("--annotated-video", action="store_true",
        help="Render annotated MP4 video(s) for pipeline inference")
    parser.add_argument("--threshold", type=float, default=0.0,
        help="Minimum prediction score for pipeline inference outputs")
    parser.set_defaults(func=run)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="trace",
        description="TRACE - Temporal Action Detection for Animal Behavior",
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
        help="Check PyPI for a newer TRACE package")
    update_parser.add_argument("--timeout", type=float, default=5.0,
        help="Seconds to wait for the PyPI version check (default: 5)")
    update_parser.set_defaults(func=update)

    train_parser = subparsers.add_parser("train", help="Train a model")
    _add_train_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    _add_eval_args(eval_parser)

    predict_parser = subparsers.add_parser("predict",
        help="Run prediction on videos (no annotations needed)")
    _add_predict_args(predict_parser)

    pipeline_parser = subparsers.add_parser("pipeline",
        help="Run the pipeline: train -> predict -> export")
    _add_pipeline_args(pipeline_parser)

    dev_parser = subparsers.add_parser("dev", help="Developer utilities")
    dev_subparsers = dev_parser.add_subparsers(dest="dev_command", metavar="COMMAND", required=True)
    dev_serve_parser = dev_subparsers.add_parser("serve",
        help="Start backend and frontend dev servers")
    _add_serve_args(dev_serve_parser, default_dev=True)
    dev_build_parser = dev_subparsers.add_parser("build-frontend",
        help="Build frontend and copy to the Python package")
    _add_build_frontend_args(dev_build_parser)

    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)

    if handler is None:
        parser.print_help()
        return None
    return handler(args)


if __name__ == "__main__":
    main()
