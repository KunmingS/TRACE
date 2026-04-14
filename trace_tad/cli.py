"""CLI entry point for TRACE.

Usage:
    trace serve              # Start server (serves bundled frontend + API)
    trace serve --dev        # Dev mode: Vite HMR + backend on separate ports
    trace build-frontend     # Build frontend and copy into package for distribution
    trace train              # Train with small model (default)
    trace train --model large  # Train with large model
    trace train --dataset-path /my/data  # Auto-clip + train (simplified)
    trace test --model-path /my/model --dataset-path /my/data  # Simplified test
    trace infer --model-path /my/model --input /my/video.mp4  # Simplified infer
    trace run <config>   # Run the full pipeline: train -> infer -> export
    trace api            # Start the pipeline API server
"""
import argparse
import os
import shutil
import subprocess
import signal
import sys
import time

from trace_tad.config import DictAction


MODEL_CONFIGS = {
    "small": "configs/tridet/tridet_small.py",
    "large": "configs/tridet/tridet_large.py",
}


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


def _resolve_model_path(args):
    """Resolve checkpoint and class_map from --model-path if provided."""
    model_path = getattr(args, "model_path", None)
    if model_path is None:
        return

    model_path = os.path.abspath(model_path)
    best_pth = os.path.join(model_path, "best.pth")
    classmap = os.path.join(model_path, "classmap.txt")

    if not os.path.isfile(best_pth):
        print(f"Error: {best_pth} not found in model path.")
        sys.exit(1)
    if not os.path.isfile(classmap):
        print(f"Error: {classmap} not found in model path.")
        sys.exit(1)

    # Set checkpoint and class_map from model path (only if not explicitly set)
    if hasattr(args, "checkpoint") and args.checkpoint is None:
        args.checkpoint = best_pth
    if hasattr(args, "class_map") and args.class_map is None:
        args.class_map = classmap


def _resolve_dataset_path(args):
    """Resolve data_path, annotation, class_map from --dataset-path if provided."""
    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path is None:
        return

    from trace_tad.data_prep import prepare_dataset

    dataset_path = os.path.abspath(dataset_path)
    print(f"Preparing dataset from: {dataset_path}")
    clips_dir, json_path, classmap_path = prepare_dataset(dataset_path)

    # Override individual flags (only if not explicitly set)
    if args.data_path is None:
        args.data_path = clips_dir
    if args.annotation is None:
        args.annotation = json_path
    if hasattr(args, "class_map") and args.class_map is None:
        args.class_map = classmap_path

    print()


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
            print("  Run 'trace build-frontend' first, or use 'trace serve --dev'.")
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
    print("Done. Run 'trace serve' to start the server.")


def train(args):
    """Run a training job."""
    from trace_tad.jobs import JobManager, TrainRequest

    # Resolve --dataset-path into individual paths
    _resolve_dataset_path(args)

    request = TrainRequest(
        config_path=_resolve_config(args),
        nproc=args.nproc,
        seed=args.seed,
        exp_id=args.id,
        resume=args.resume,
        not_eval=args.not_eval,
        disable_deterministic=args.disable_deterministic,
        dataset_dir=args.data_path,
        annotation_path=args.annotation,
        class_map=args.class_map,
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
    else:
        print(f"\nTraining {job.status.value}: {job.error_message or ''}")
    sys.exit(job.return_code or (0 if job.status.value == "completed" else 1))


def test(args):
    """Run a test/inference job."""
    from trace_tad.jobs import JobManager, TestRequest

    # Resolve --model-path and --dataset-path
    _resolve_model_path(args)
    _resolve_dataset_path(args)

    if args.checkpoint is None:
        print("Error: --checkpoint or --model-path is required.")
        sys.exit(1)

    # Warn if model-path is set but no dataset specified
    if getattr(args, "model_path", None) and not getattr(args, "dataset_path", None) \
            and args.data_path is None and args.annotation is None:
        print("Warning: --model-path set but no dataset specified (--dataset-path or --data-path).")
        print("  Using config defaults for data. Pass --dataset-path to override.")
        print()

    request = TestRequest(
        config_path=_resolve_config(args),
        checkpoint=args.checkpoint,
        nproc=args.nproc,
        seed=args.seed,
        exp_id=args.id,
        not_eval=args.not_eval,
        profile=args.profile,
        auto_tune=args.auto_tune,
        dataset_dir=args.data_path,
        annotation_path=args.annotation,
        class_map=args.class_map,
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
    from trace_tad.jobs import JobManager, InferRequest

    # Resolve --model-path
    _resolve_model_path(args)

    if args.checkpoint is None:
        print("Error: --checkpoint or --model-path is required.")
        sys.exit(1)
    if args.class_map is None:
        print("Error: --class-map or --model-path is required.")
        sys.exit(1)

    request = InferRequest(
        config_path=_resolve_config(args),
        checkpoint=args.checkpoint,
        input=args.input,
        class_map=args.class_map,
        output=args.output,
        seed=args.seed,
        exp_id=args.id,
        profile=args.profile,
        auto_tune=args.auto_tune,
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


def run(args):
    """Run the TRACE pipeline: train -> infer -> export."""
    from trace_tad.pipeline import run_pipeline

    # Determine mode
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


def api(args):
    """Start the TRACE pipeline API server."""
    print(f"Starting pipeline API server on {args.host}:{args.port}...")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "trace_tad.api:app",
            "--host", args.host,
            "--port", str(args.port),
            "--reload",
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        prog="trace",
        description="TRACE - Temporal Action Detection for Animal Behavior",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start annotation server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    serve_parser.add_argument("--dev", action="store_true",
        help="Development mode: start Vite dev server + backend separately (requires Node.js)")
    serve_parser.add_argument("--frontend-port", type=int, default=3000,
        help="Frontend port in --dev mode (default: 3000)")
    serve_parser.add_argument("--backend-only", action="store_true",
        help="Only start backend (--dev mode only)")
    serve_parser.add_argument("--frontend-only", action="store_true",
        help="Only start frontend (--dev mode only)")

    # build-frontend command
    build_parser = subparsers.add_parser("build-frontend",
        help="Build frontend and copy to package for distribution")
    build_parser.add_argument("--skip-npm", action="store_true",
        help="Skip npm build, just copy existing dist/")

    # train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, choices=["small", "large"], default="small",
        help="Model size (default: small)")
    train_parser.add_argument("--config", type=str, default=None,
        help="Custom config file path (overrides --model)")
    train_parser.add_argument("--dataset-path", type=str, default=None,
        help="Dataset directory with videos+CSVs (auto-clips and generates annotations)")
    train_parser.add_argument("--data-path", type=str, default=None,
        help="Video clip directory (overrides config's data_path)")
    train_parser.add_argument("--annotation", type=str, default=None,
        help="Annotation JSON path (overrides config's annotation_path)")
    train_parser.add_argument("--class-map", type=str, default=None,
        help="Class map file (overrides config; auto-generated from annotations if omitted)")
    train_parser.add_argument("--pretrained", type=str, default=None,
        help="Pretrained backbone weights path (overrides config's pretrain)")
    train_parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs (default: 1)")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    train_parser.add_argument("--id", type=int, default=0, help="Experiment repeat ID (default: 0)")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    train_parser.add_argument("--not-eval", action="store_true", help="Skip evaluation, inference only")
    train_parser.add_argument("--disable-deterministic", action="store_true",
        help="Disable deterministic for faster speed")
    train_parser.add_argument("--cfg-options", nargs="+", action=DictAction,
        help="Override config settings (key=value pairs)")

    # test command
    test_parser = subparsers.add_parser("test", help="Test/evaluate a model")
    test_parser.add_argument("--model", type=str, choices=["small", "large"], default="small",
        help="Model size (default: small)")
    test_parser.add_argument("--config", type=str, default=None,
        help="Custom config file path (overrides --model)")
    test_parser.add_argument("--model-path", type=str, default=None,
        help="Model directory containing best.pth and classmap.txt")
    test_parser.add_argument("--checkpoint", type=str, default=None,
        help="Checkpoint path (required if --model-path not set)")
    test_parser.add_argument("--dataset-path", type=str, default=None,
        help="Dataset directory with videos+CSVs (auto-clips and generates annotations)")
    test_parser.add_argument("--data-path", type=str, default=None,
        help="Video clip directory (overrides config's data_path)")
    test_parser.add_argument("--annotation", type=str, default=None,
        help="Annotation JSON path (overrides config's annotation_path)")
    test_parser.add_argument("--class-map", type=str, default=None,
        help="Class map file (overrides config; auto-generated from annotations if omitted)")
    test_parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs (default: 1)")
    test_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    test_parser.add_argument("--id", type=int, default=0, help="Experiment repeat ID (default: 0)")
    test_parser.add_argument("--not-eval", action="store_true", help="Skip evaluation, inference only")
    test_parser.add_argument("--profile", action="store_true",
        help="Enable inference profiling (CPU + GPU timing breakdown)")
    test_parser.add_argument("--auto-tune", action=argparse.BooleanOptionalAction, default=True,
        help="Auto-tune dataloader params (default: enabled, use --no-auto-tune to disable)")
    test_parser.add_argument("--cfg-options", nargs="+", action=DictAction,
        help="Override config settings (key=value pairs)")

    # infer command
    infer_parser = subparsers.add_parser("infer",
        help="Run inference on videos (no annotations needed)")
    infer_parser.add_argument("--model", type=str, choices=["small", "large"], default="small",
        help="Model size (default: small)")
    infer_parser.add_argument("--config", type=str, default=None,
        help="Custom config file path (overrides --model)")
    infer_parser.add_argument("--model-path", type=str, default=None,
        help="Model directory containing best.pth and classmap.txt")
    infer_parser.add_argument("--checkpoint", type=str, default=None,
        help="Model checkpoint path (required if --model-path not set)")
    infer_parser.add_argument("--input", type=str, required=True,
        help="Input video file or directory of videos")
    infer_parser.add_argument("--class-map", type=str, default=None,
        help="Class map file (required if --model-path not set)")
    infer_parser.add_argument("--output", type=str, default=None,
        help="Output JSON path (default: predictions.json in work_dir)")
    infer_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    infer_parser.add_argument("--id", type=int, default=0, help="Experiment repeat ID (default: 0)")
    infer_parser.add_argument("--profile", action="store_true",
        help="Enable inference profiling (CPU + GPU timing breakdown)")
    infer_parser.add_argument("--auto-tune", action=argparse.BooleanOptionalAction, default=True,
        help="Auto-tune dataloader params (default: enabled, use --no-auto-tune to disable)")
    infer_parser.add_argument("--cfg-options", nargs="+", action=DictAction,
        help="Override config settings (key=value pairs)")

    # run command (pipeline orchestrator)
    run_parser = subparsers.add_parser("run", help="Run the pipeline: train -> infer -> export")
    run_parser.add_argument("config", metavar="FILE", type=str, help="Path to config file")
    run_parser.add_argument("--train-only", action="store_true", help="Only run training")
    run_parser.add_argument("--infer-only", action="store_true", help="Only run inference (requires --checkpoint)")
    run_parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for inference")
    run_parser.add_argument("--infer-videos", nargs="+", default=None, help="Video file paths for inference")
    run_parser.add_argument("--export", choices=["csv"], default=None, help="Export results format")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="Override config settings")

    # api command
    api_parser = subparsers.add_parser("api", help="Start the pipeline API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8001, help="API port (default: 8001)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "serve":
        serve(args)
    elif args.command == "build-frontend":
        build_frontend(args)
    elif args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "run":
        run(args)
    elif args.command == "api":
        api(args)


if __name__ == "__main__":
    main()
