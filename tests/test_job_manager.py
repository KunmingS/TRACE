import sys
from pathlib import Path

import trace_tad.jobs.manager as manager_mod
from trace_tad.jobs.models import InferRequest, JobStatus, PrepRequest, TestRequest as EvalRequest, TrainRequest


def _build_manager(monkeypatch, tmp_path):
    trace_home = tmp_path / ".trace-home"
    monkeypatch.setattr(manager_mod, "_find_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(manager_mod, "_trace_home", lambda: trace_home)
    return manager_mod.JobManager(max_concurrency=1)


def test_build_train_command_includes_shortcuts_and_cfg_options(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)
    request = TrainRequest(
        config_path="configs/tridet/tridet_small.py",
        nproc=2,
        seed=7,
        exp_id=31,
        resume="/tmp/checkpoint.pth",
        not_eval=True,
        disable_deterministic=True,
        dataset_dir="/data/clips",
        annotation_path="/data/dataset.json",
        class_map="/data/classmap.txt",
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
    assert "configs/tridet/tridet_small.py" in command
    assert "--resume" in command
    assert "--not_eval" in command
    assert "--disable_deterministic" in command
    cfg_options = command[command.index("--cfg-options") + 1:]
    assert "data_path=/data/clips" in cfg_options
    assert "annotation_path=/data/dataset.json" in cfg_options
    assert "class_map=/data/classmap.txt" in cfg_options
    assert "model.projection.custom.pretrain=/weights/pretrain.pth" in cfg_options
    assert "solver.lr=0.001" in cfg_options
    assert "workflow.logging_interval=25" in cfg_options


def test_build_test_infer_and_prep_commands_cover_new_job_types(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)

    test_command = manager._build_test_command(
        EvalRequest(
            config_path="configs/tridet/tridet_large.py",
            checkpoint="/models/best.pth",
            nproc=1,
            seed=11,
            exp_id=22,
            not_eval=False,
            profile=True,
            auto_tune=True,
            dataset_dir="/prepared/clips",
            annotation_path="/prepared/dataset.json",
            class_map="/prepared/classmap.txt",
            cfg_options={"evaluation.top_k": 5},
        )
    )
    infer_command = manager._build_infer_command(
        InferRequest(
            config_path="configs/tridet/tridet_small.py",
            checkpoint="/models/best.pth",
            input="/videos/input.mp4",
            class_map="/models/classmap.txt",
            output="/tmp/predictions.json",
            exp_id=44,
            profile=True,
            auto_tune=True,
            cfg_options={"inference.score_thresh": 0.6},
        )
    )
    prep_command = manager._build_prep_command(
        PrepRequest(
            dataset_path="/datasets/raw",
            clip_frames=512,
            train_ratio=0.7,
        )
    )

    assert test_command[:3] == [sys.executable, "tools/test.py", "configs/tridet/tridet_large.py"]
    assert "--checkpoint" in test_command
    assert "--profile" in test_command
    assert "--auto-tune" in test_command
    test_cfg_options = test_command[test_command.index("--cfg-options") + 1:]
    assert "data_path=/prepared/clips" in test_cfg_options
    assert "annotation_path=/prepared/dataset.json" in test_cfg_options
    assert "class_map=/prepared/classmap.txt" in test_cfg_options
    assert "evaluation.top_k=5" in test_cfg_options

    assert infer_command[:3] == [sys.executable, "tools/infer.py", "configs/tridet/tridet_small.py"]
    assert "--input" in infer_command
    assert "--class-map" in infer_command
    assert "--output" in infer_command
    assert "--profile" in infer_command
    assert "--auto-tune" in infer_command
    infer_cfg_options = infer_command[infer_command.index("--cfg-options") + 1:]
    assert "inference.score_thresh=0.6" in infer_cfg_options

    assert prep_command == [
        sys.executable,
        "tools/prep_dataset.py",
        "/datasets/raw",
        "--clip-frames",
        "512",
        "--train-ratio",
        "0.7",
    ]


def test_start_prep_job_runs_subprocess_and_sets_work_dir(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)
    script = (
        "from pathlib import Path; "
        "Path('prep_result.json').write_text('{\"ok\": true}'); "
        "print('prepared dataset')"
    )
    monkeypatch.setattr(
        manager_mod.JobManager,
        "_build_prep_command",
        lambda self, request: [sys.executable, "-c", script],
    )

    job = manager.start_prep_job(PrepRequest(dataset_path="/datasets/raw"))
    completed = manager.wait_for_job(job.job_id)

    assert completed.status == JobStatus.COMPLETED
    assert completed.work_dir == str(tmp_path)
    assert Path(completed.log_file).read_text(encoding="utf-8").strip() == "prepared dataset"
    assert (tmp_path / "prep_result.json").read_text(encoding="utf-8") == "{\"ok\": true}"
