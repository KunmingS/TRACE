import pytest

from trace_tad import cli
from trace_tad.weights import model_weight_choices, model_weight_names


def test_predict_dispatches_to_infer(monkeypatch):
    calls = []

    def fake_infer(args):
        calls.append((args.command, args.model_dir, args.input, args.annotated_video, args.threshold))

    monkeypatch.setattr(cli, "infer", fake_infer)

    cli.main([
        "predict",
        "--model-dir", "/models/model_20260507_120000",
        "--input", "/videos",
        "--annotated-video",
        "--threshold", "0.42",
    ])
    assert calls == [("predict", "/models/model_20260507_120000", "/videos", True, 0.42)]


def test_eval_accepts_pair_selection(monkeypatch):
    calls = []

    def fake_eval(args):
        calls.append((args.command, args.model_dir, args.work_dir, args.explicit_pairs))

    monkeypatch.setattr(cli, "test", fake_eval)

    cli.main([
        "eval",
        "--model-dir", "/models/model_20260507_120000",
        "--work-dir", "/datasets/new",
        "--pairs", "mouse01.mp4=mouse01.csv",
    ])
    assert calls == [(
        "eval",
        "/models/model_20260507_120000",
        "/datasets/new",
        ["mouse01.mp4=mouse01.csv"],
    )]


def test_train_accepts_pair_selection(monkeypatch):
    calls = []

    def fake_train(args):
        calls.append((args.work_dir, args.explicit_pairs))

    monkeypatch.setattr(cli, "train", fake_train)

    cli.main([
        "train",
        "--work-dir", "/datasets/train",
        "--pairs", "alpha.mp4=alpha.csv", "beta.mp4=beta.csv",
    ])

    assert calls == [(
        "/datasets/train",
        ["alpha.mp4=alpha.csv", "beta.mp4=beta.csv"],
    )]


def test_train_requires_explicit_pairs():
    with pytest.raises(SystemExit):
        cli.main(["train", "--work-dir", "/datasets/train"])


def test_pipeline_legacy_config_mode_still_dispatches(monkeypatch):
    calls = []

    class State:
        phase = "completed"
        error = None

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        return State()

    import trace_tad.pipeline as pipeline_mod

    monkeypatch.setattr(cli, "_require_cuda", lambda: None)
    monkeypatch.setattr(pipeline_mod, "run_pipeline", fake_run_pipeline)

    cli.main(["pipeline", "configs/small.py", "--train-only"])

    assert calls == [{
        "config": "configs/small.py",
        "mode": "train_only",
        "checkpoint": None,
        "infer_videos": [],
        "export_format": None,
        "cfg_options": {},
        "seed": 42,
    }]


def test_pipeline_ui_flag_mode_builds_shared_spec(monkeypatch):
    calls = []

    def fake_run_pipeline_spec(spec):
        calls.append(spec)

    monkeypatch.setattr(cli, "_run_pipeline_spec", fake_run_pipeline_spec)

    cli.main([
        "pipeline",
        "--model", "large",
        "--train",
        "--work-dir", "/datasets/train",
        "--pairs", "alpha.mp4=alpha.csv",
        "--cache-mode", "virtual",
        "--cache-resolution", "192",
        "--train-ratio", "0.7",
        "--epochs", "60",
        "--val-start-epoch", "10",
        "--val-interval", "3",
        "--resource-profile", "low",
        "--train-workers", "3",
        "--train-decode-threads", "2",
        "--train-prefetch", "4",
        "--infer",
        "--infer-resource-profile", "high",
        "--infer-batch-size", "12",
        "--infer-workers", "10",
        "--infer-decode-threads", "3",
        "--infer-prefetch", "4",
        "--input", "/videos",
        "--include-stems", "clip01", "clip02",
        "--annotated-video",
        "--threshold", "0.37",
    ])

    assert len(calls) == 1
    spec = calls[0]
    assert spec.model_size == "large"
    assert spec.steps.train is True
    assert spec.steps.infer is True
    assert spec.train_selection.folder == "/datasets/train"
    assert spec.train_selection.pairs == ["alpha.mp4=alpha.csv"]
    assert spec.cache_mode == "virtual"
    assert spec.cache_resolution == 192
    assert spec.train_ratio == 0.7
    assert spec.epochs == 60
    assert spec.val_start_epoch == 10
    assert spec.val_interval == 3
    assert spec.resource_profile == "low"
    assert spec.resources.train.profile == "low"
    assert spec.resources.train.num_workers == 3
    assert spec.resources.train.decode_threads == 2
    assert spec.resources.train.prefetch_factor == 4
    assert spec.resources.infer.profile == "high"
    assert spec.resources.infer.batch_size == 12
    assert spec.resources.infer.num_workers == 10
    assert spec.resources.infer.decode_threads == 3
    assert spec.resources.infer.prefetch_factor == 4
    assert spec.input_selection.folder == "/videos"
    assert spec.input_selection.stems == ["clip01", "clip02"]
    assert spec.annotated_video is True
    assert spec.threshold == 0.37


def test_legacy_subcommands_are_removed():
    for command in ("serve", "test", "infer", "run", "download-weights", "build-frontend", "dataset", "api"):
        with pytest.raises(SystemExit):
            cli.main([command, "--help"])


def test_prepare_downloads_selected_model_weights(monkeypatch):
    calls = []

    def fake_download(selection):
        calls.append(selection)
        return [f"/cache/{selection}.pth"]

    monkeypatch.setattr(cli, "_download_weights_selection", fake_download)

    result = cli.main(["prepare", "--weights", "small"])

    assert calls == ["small"]
    assert result == ["/cache/small.pth"]


def test_update_reports_up_to_date(monkeypatch, capsys):
    monkeypatch.setattr(cli, "__version__", "0.2.0")
    monkeypatch.setattr(cli, "_fetch_latest_pypi_version", lambda timeout: "0.2.0")

    result = cli.main(["update"])

    captured = capsys.readouterr()
    assert result == 0
    assert "TRACE is up to date (0.2.0)." in captured.out


def test_update_prompts_when_pypi_has_newer_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "__version__", "0.2.0")
    monkeypatch.setattr(cli, "_fetch_latest_pypi_version", lambda timeout: "0.3.0")

    result = cli.main(["update"])

    captured = capsys.readouterr()
    assert result == 0
    assert "TRACE 0.3.0 is available on PyPI." in captured.out
    assert "Installed version: 0.2.0" in captured.out
    # Non-interactive (no TTY in tests): prints the manual command, never hangs.
    assert "python -m pip install --upgrade trace-tad" in captured.out


class _FakeTTY:
    def isatty(self):
        return True


def _stub_update_env(monkeypatch, latest="0.3.0"):
    monkeypatch.setattr(cli, "__version__", "0.2.0")
    monkeypatch.setattr(cli, "_fetch_latest_pypi_version", lambda timeout: latest)
    monkeypatch.setattr(cli, "_ensure_ffmpeg", lambda: None)


def test_update_check_only_never_installs(monkeypatch, capsys):
    _stub_update_env(monkeypatch)

    def _boom():
        raise AssertionError("--check-only must not install")

    monkeypatch.setattr(cli, "_run_pip_upgrade", _boom)
    result = cli.main(["update", "--check-only"])
    out = capsys.readouterr().out
    assert result == 0
    assert "python -m pip install --upgrade trace-tad" in out


def test_update_yes_installs_without_prompt(monkeypatch, capsys):
    _stub_update_env(monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "_run_pip_upgrade", lambda: calls.append(True) or 0)
    result = cli.main(["update", "--yes"])
    out = capsys.readouterr().out
    assert result == 0 and calls == [True]
    assert "Updated to trace-tad 0.3.0" in out


def test_update_interactive_yes_installs(monkeypatch, capsys):
    _stub_update_env(monkeypatch)
    monkeypatch.setattr("sys.stdin", _FakeTTY())
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    calls = []
    monkeypatch.setattr(cli, "_run_pip_upgrade", lambda: calls.append(True) or 0)
    result = cli.main(["update"])
    assert result == 0 and calls == [True]


def test_update_interactive_no_skips_install(monkeypatch, capsys):
    _stub_update_env(monkeypatch)
    monkeypatch.setattr("sys.stdin", _FakeTTY())
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    def _boom():
        raise AssertionError("answering n must not install")

    monkeypatch.setattr(cli, "_run_pip_upgrade", _boom)
    result = cli.main(["update"])
    out = capsys.readouterr().out
    assert result == 0
    assert "python -m pip install --upgrade trace-tad" in out


def test_update_yes_propagates_pip_failure(monkeypatch, capsys):
    _stub_update_env(monkeypatch)
    monkeypatch.setattr(cli, "_run_pip_upgrade", lambda: 1)
    result = cli.main(["update", "--yes"])
    out = capsys.readouterr().out
    assert result == 1
    assert "Update failed" in out


def test_update_reports_pypi_check_failure(monkeypatch, capsys):
    def fake_fetch(timeout):
        raise RuntimeError("Could not reach PyPI: offline")

    monkeypatch.setattr(cli, "_fetch_latest_pypi_version", fake_fetch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["update"])

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "Could not reach PyPI: offline" in captured.err


def test_model_weight_selection_helpers():
    assert model_weight_choices() == ("all", "small", "large")
    assert model_weight_names("small") == [
        "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
    ]
    assert model_weight_names("all") == [
        "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
        "vit-large-p16_videomaev2-k400.pth",
    ]

    with pytest.raises(ValueError, match="Unknown model weight selection"):
        model_weight_names("medium")
