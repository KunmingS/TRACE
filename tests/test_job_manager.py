import sys
from pathlib import Path

import trace_tad.jobs.manager as manager_mod
from trace_tad.model_artifacts import create_predict_dir_for_input
from trace_tad.jobs.models import (
    InferRequest,
    JobStatus,
    PrepRequest,
    TestRequest as EvalRequest,
    TrainRequest,
    TrainTuneRequest,
)


def _build_manager(monkeypatch, tmp_path):
    trace_home = tmp_path / ".trace-home"
    monkeypatch.setattr(manager_mod, "_find_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(manager_mod, "_trace_home", lambda: trace_home)
    return manager_mod.JobManager(max_concurrency=1)


def test_build_train_command_includes_shortcuts_and_cfg_options(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)
    request = TrainRequest(
        config_path="configs/small.py",
        model_dir="/runs/model_20260507_120000",
        nproc=2,
        seed=7,
        resume="/tmp/checkpoint.pth",
        not_eval=True,
        disable_deterministic=True,
        dataset_dir="/data/model_20260507_120000",
        annotation_path="/data/model_20260507_120000/dataset.json",
        class_map="/data/model_20260507_120000/classmap.txt",
        pretrained="/weights/pretrain.pth",
        cfg_options={"solver.lr": 0.001, "workflow.logging_interval": 25},
    )

    command = manager._build_train_command(request)

    assert command[:6] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "tools/train.py",
    ]
    assert "configs/small.py" in command
    assert "--resume" in command
    assert "--not_eval" in command
    assert "--disable_deterministic" in command
    cfg_options = command[command.index("--cfg-options") + 1:]
    assert "work_dir=/runs/model_20260507_120000" in cfg_options
    assert "data_path=/data/model_20260507_120000" in cfg_options
    assert "annotation_path=/data/model_20260507_120000/dataset.json" in cfg_options
    assert "class_map=/data/model_20260507_120000/classmap.txt" in cfg_options
    assert "model.projection.custom.pretrain=/weights/pretrain.pth" in cfg_options
    assert "solver.lr=0.001" in cfg_options
    assert "workflow.logging_interval=25" in cfg_options


def test_build_test_infer_and_prep_commands_cover_new_job_types(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)

    test_command = manager._build_test_command(
        EvalRequest(
            model_dir="/models/model_20260507_120000",
            config_path="configs/large.py",
            checkpoint="/models/best.pth",
            output_dir="/models/model_20260507_120000/eval_20260507_121000",
            nproc=1,
            seed=11,
            not_eval=False,
            profile=True,
            auto_tune=True,
            dataset_dir="/models/model_20260507_120000",
            annotation_path="/models/model_20260507_120000/dataset.json",
            class_map="/models/model_20260507_120000/classmap.txt",
            cfg_options={"evaluation.top_k": 5},
        )
    )
    infer_command = manager._build_infer_command(
        InferRequest(
            model_dir="/models/model_20260507_120000",
            config_path="configs/small.py",
            checkpoint="/models/best.pth",
            input="/videos/input.mp4",
            class_map="/models/classmap.txt",
            output="/tmp/predictions.json",
            output_dir="/models/model_20260507_120000/predict_20260507_121000",
            profile=True,
            auto_tune=True,
            annotated_video=True,
            threshold=0.6,
            cfg_options={"inference.score_thresh": 0.6},
        )
    )
    prep_command = manager._build_prep_command(
        PrepRequest(
            work_dir="/datasets/raw",
            model_dir="/datasets/raw/model_20260507_120000",
            clip_frames=512,
            train_ratio=0.7,
            cache_mode="virtual",
        )
    )
    tune_command = manager._build_train_tune_command(
        TrainTuneRequest(
            config_path="configs/small.py",
            model_dir="/datasets/raw/model_20260507_120000",
            annotation_path="/datasets/raw/model_20260507_120000/dataset.json",
            class_map="/datasets/raw/model_20260507_120000/classmap.txt",
        )
    )

    assert test_command[:3] == [sys.executable, "tools/test.py", "configs/large.py"]
    assert "--checkpoint" in test_command
    assert "--profile" in test_command
    assert "--auto-tune" in test_command
    test_cfg_options = test_command[test_command.index("--cfg-options") + 1:]
    assert "data_path=/models/model_20260507_120000" in test_cfg_options
    assert "annotation_path=/models/model_20260507_120000/dataset.json" in test_cfg_options
    assert "class_map=/models/model_20260507_120000/classmap.txt" in test_cfg_options
    assert "work_dir=/models/model_20260507_120000/eval_20260507_121000" in test_cfg_options
    assert "evaluation.top_k=5" in test_cfg_options

    assert infer_command[:3] == [sys.executable, "tools/infer.py", "configs/small.py"]
    assert "--input" in infer_command
    assert "--class-map" in infer_command
    assert "--output" in infer_command
    assert "--annotated-video" in infer_command
    assert "--threshold" in infer_command
    assert infer_command[infer_command.index("--threshold") + 1] == "0.6"
    assert "--profile" in infer_command
    assert "--auto-tune" in infer_command
    infer_cfg_options = infer_command[infer_command.index("--cfg-options") + 1:]
    assert "work_dir=/models/model_20260507_120000/predict_20260507_121000" in infer_cfg_options
    assert "inference.score_thresh=0.6" in infer_cfg_options

    assert prep_command == [
        sys.executable,
        "tools/prep_dataset.py",
        "/datasets/raw",
        "--clip-frames",
        "512",
        "--train-ratio",
        "0.7",
        "--cache-mode",
        "virtual",
        "--cache-resolution",
        "144",
        "--cache-crf",
        "23",
        "--output-dir",
        "/datasets/raw/model_20260507_120000",
        "--output",
        "/datasets/raw/model_20260507_120000/prep_result.json",
    ]
    assert tune_command == [
        sys.executable,
        "tools/tune_train.py",
        "configs/small.py",
        "--model-dir",
        "/datasets/raw/model_20260507_120000",
        "--annotation-path",
        "/datasets/raw/model_20260507_120000/dataset.json",
        "--class-map",
        "/datasets/raw/model_20260507_120000/classmap.txt",
        "--output",
        "/datasets/raw/model_20260507_120000/train_tune_result.json",
    ]

    # included_stems passthrough — appended to both prep and infer commands.
    prep_with_stems = manager._build_prep_command(
        PrepRequest(
            work_dir="/datasets/raw",
            model_dir="/datasets/raw/model_20260507_120000",
            included_stems=["clip01", "clip02"],
        )
    )
    assert prep_with_stems[-3:] == ["--include-stems", "clip01", "clip02"]

    infer_with_stems = manager._build_infer_command(
        InferRequest(
            model_dir="/models/model_20260507_120000",
            config_path="configs/small.py",
            checkpoint="/models/best.pth",
            input="/videos",
            class_map="/models/classmap.txt",
            output_dir="/models/model_20260507_120000/predict_20260507_121000",
            included_stems=["videoA"],
        )
    )
    assert "--include-stems" in infer_with_stems
    sidx = infer_with_stems.index("--include-stems")
    assert infer_with_stems[sidx + 1] == "videoA"


def test_predict_dir_defaults_to_input_location(tmp_path):
    input_dir = tmp_path / "videos"
    input_dir.mkdir()
    video = input_dir / "clip01.mp4"
    video.write_bytes(b"")

    from_file = Path(create_predict_dir_for_input(video))
    from_dir = Path(create_predict_dir_for_input(input_dir))

    assert from_file.parent == input_dir
    assert from_file.name.startswith("predict_")
    assert from_dir.parent == input_dir
    assert from_dir.name.startswith("predict_")


def test_start_infer_job_uses_input_location_for_work_dir(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)
    model_dir = tmp_path / "models" / "model_20260507_120000"
    model_dir.mkdir(parents=True)
    (model_dir / "best.pth").write_bytes(b"")
    (model_dir / "classmap.txt").write_text("behavior\n", encoding="utf-8")
    (model_dir / "config.txt").write_text("configs/small.py\n", encoding="utf-8")

    input_dir = tmp_path / "video-pairs"
    input_dir.mkdir()

    def fake_infer_command(self, request):
        script = (
            "from pathlib import Path; "
            f"Path({str(Path(request.output_dir) / 'predictions.json')!r}).write_text('{{\"ok\": true}}'); "
            "print('prediction ready')"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(
        manager_mod.JobManager,
        "_build_infer_command",
        fake_infer_command,
    )

    job = manager.start_infer_job(InferRequest(
        model_dir=str(model_dir),
        input=str(input_dir),
    ))
    completed = manager.wait_for_job(job.job_id)

    work_dir = Path(completed.work_dir)
    assert completed.status == JobStatus.COMPLETED
    assert work_dir.parent == input_dir
    assert work_dir.name.startswith("predict_")
    assert Path(completed.log_file) == work_dir / "job.log"
    assert (work_dir / "predictions.json").read_text(encoding="utf-8") == "{\"ok\": true}"


def test_pipeline_jobs_share_backend_run_metadata(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)

    def fake_prep_command(self, request):
        script = (
            "from pathlib import Path; "
            f"Path({str(Path(request.model_dir) / 'prep_result.json')!r}).write_text("
            "'{\"model_dir\":\"%s\",\"dataset_json\":\"dataset.json\",\"classmap_path\":\"classmap.txt\"}'"
            f" % {str(request.model_dir)!r}); "
            "print('prep ready')"
        )
        return [sys.executable, "-c", script]

    def fake_train_command(self, request):
        return [sys.executable, "-c", "print('train ready')"]

    monkeypatch.setattr(manager_mod.JobManager, "_build_prep_command", fake_prep_command)
    monkeypatch.setattr(manager_mod.JobManager, "_build_train_command", fake_train_command)

    prep = manager.start_prep_job(PrepRequest(
        work_dir=str(tmp_path),
        run_steps=["train"],
    ))
    completed_prep = manager.wait_for_job(prep.job_id)

    train = manager.start_train_job(TrainRequest(
        config_path="configs/small.py",
        model_dir=completed_prep.work_dir,
        dataset_dir=completed_prep.work_dir,
        annotation_path=str(Path(completed_prep.work_dir) / "dataset.json"),
        class_map=str(Path(completed_prep.work_dir) / "classmap.txt"),
        run_id=completed_prep.run_id,
        run_steps=["train"],
    ))
    completed_train = manager.wait_for_job(train.job_id)

    assert completed_prep.run_id == completed_prep.job_id
    assert completed_train.run_id == completed_prep.run_id
    assert completed_prep.stage == "prep"
    assert completed_train.stage == "train"
    assert completed_prep.run_steps == ["train"]
    assert completed_train.run_steps == ["train"]
    assert "run_id" not in completed_train.args
    assert "run_steps" not in completed_train.args


def test_start_prep_job_runs_subprocess_and_sets_work_dir(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)
    def fake_prep_command(self, request):
        script = (
            "from pathlib import Path; "
            f"Path({str(Path(request.model_dir) / 'prep_result.json')!r}).write_text('{{\"ok\": true}}'); "
            "print('model dataset ready')"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(
        manager_mod.JobManager,
        "_build_prep_command",
        fake_prep_command,
    )

    job = manager.start_prep_job(PrepRequest(work_dir=str(tmp_path)))
    completed = manager.wait_for_job(job.job_id)

    assert completed.status == JobStatus.COMPLETED
    assert Path(completed.work_dir).name.startswith("model_")
    assert Path(completed.log_file).read_text(encoding="utf-8").strip() == "model dataset ready"
    assert (Path(completed.work_dir) / "prep_result.json").read_text(encoding="utf-8") == "{\"ok\": true}"
