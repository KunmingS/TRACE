import pytest

from trace_tad.pipeline_plan import (
    PipelineResources,
    PipelineResourceSettings,
    PipelineSelection,
    PipelineSpec,
    PipelineSpecError,
    PipelineSteps,
    build_pipeline_command,
)


def test_build_train_only_command_quotes_paths_and_pairs():
    spec = PipelineSpec(
        steps=PipelineSteps(train=True),
        model_size="large",
        train_selection=PipelineSelection(
            folder="/data/dev set",
            pairs=["/data/dev set/video 01.mp4=/data/dev set/labels 01.csv"],
        ),
        cache_mode="virtual",
        resource_profile="balanced",
        epochs=80,
        val_start_epoch=20,
        val_interval=5,
        train_ratio=0.75,
    )

    result = build_pipeline_command(spec)

    assert result.argv == [
        "trace", "pipeline", "--model", "large",
        "--train",
        "--work-dir", "/data/dev set",
        "--pairs", "/data/dev set/video 01.mp4=/data/dev set/labels 01.csv",
        "--cache-mode", "virtual",
        "--cache-resolution", "144",
        "--train-ratio", "0.75",
        "--epochs", "80",
        "--val-start-epoch", "20",
        "--val-interval", "5",
        "--resource-profile", "balanced",
    ]
    assert "'/data/dev set'" in result.command
    assert "'/data/dev set/video 01.mp4=/data/dev set/labels 01.csv'" in result.command


def test_build_train_and_infer_command_round_trips_selection():
    spec = PipelineSpec(
        steps=PipelineSteps(train=True, infer=True),
        train_selection=PipelineSelection(folder="/data/train", pairs=["a.mp4=a.csv"]),
        input_selection=PipelineSelection(folder="/videos/new", stems=["clip01", "clip02"]),
    )

    result = build_pipeline_command(spec)

    assert "--train" in result.argv
    assert "--infer" in result.argv
    assert result.argv[result.argv.index("--input") + 1] == "/videos/new"
    assert result.argv[-2:] == ["clip01", "clip02"]


def test_build_eval_only_command_requires_model_and_test_pairs():
    spec = PipelineSpec(
        steps=PipelineSteps(extra_test=True),
        model_dir="/models/model_20260507_120000",
        test_selection=PipelineSelection(folder="/data/test", pairs=["test.mp4=test.csv"]),
        cache_mode="virtual",
    )

    result = build_pipeline_command(spec)

    assert result.argv[:6] == [
        "trace", "pipeline", "--model", "small", "--extra-test", "--model-dir",
    ]
    assert "/models/model_20260507_120000" in result.argv
    assert "test.mp4=test.csv" in result.argv
    assert result.argv[result.argv.index("--cache-mode") + 1] == "cached_video"
    assert result.argv[result.argv.index("--cache-resolution") + 1] == "144"


def test_build_infer_only_command_requires_model_and_stems():
    spec = PipelineSpec(
        steps=PipelineSteps(infer=True),
        model_dir="/models/model_20260507_120000",
        input_selection=PipelineSelection(folder="/videos", stems=["videoA"]),
        annotated_video=True,
        threshold=0.35,
    )

    result = build_pipeline_command(spec)

    assert result.argv == [
        "trace", "pipeline", "--model", "small",
        "--infer",
        "--model-dir", "/models/model_20260507_120000",
        "--infer-resource-profile", "balanced",
        "--input", "/videos",
        "--include-stems", "videoA",
        "--annotated-video",
        "--threshold", "0.35",
    ]


def test_train_extra_test_reuses_train_selection():
    spec = PipelineSpec(
        steps=PipelineSteps(train=True, extra_test=True),
        train_selection=PipelineSelection(folder="/data/train", pairs=["train.mp4=train.csv"]),
        test_selection=PipelineSelection(folder="/data/test", pairs=["test.mp4=test.csv"]),
    )

    result = build_pipeline_command(spec)

    assert "train.mp4=train.csv" in result.argv
    assert "test.mp4=test.csv" not in result.argv
    assert "--extra-test" in result.argv
    assert result.argv[result.argv.index("--cache-mode") + 1] == "cached_video"


def test_build_command_includes_advanced_resource_overrides():
    spec = PipelineSpec(
        steps=PipelineSteps(train=True, extra_test=True, infer=True),
        train_selection=PipelineSelection(folder="/data/train", pairs=["train.mp4=train.csv"]),
        input_selection=PipelineSelection(folder="/videos", stems=["clip"]),
        resources=PipelineResources(
            train=PipelineResourceSettings(profile="balanced", batch_size=4, num_workers=6, decode_threads=3, prefetch_factor=4),
            test=PipelineResourceSettings(profile="low", batch_size=2, num_workers=3, decode_threads=1, prefetch_factor=2),
            infer=PipelineResourceSettings(profile="high", batch_size=12, num_workers=10, decode_threads=3, prefetch_factor=4),
        ),
    )

    result = build_pipeline_command(spec)

    assert "--train-workers" in result.argv
    assert result.argv[result.argv.index("--train-workers") + 1] == "6"
    assert "--train-decode-threads" in result.argv
    assert "--train-prefetch" in result.argv
    assert result.argv[result.argv.index("--test-resource-profile") + 1] == "low"
    assert result.argv[result.argv.index("--test-batch-size") + 1] == "2"
    assert result.argv[result.argv.index("--test-workers") + 1] == "3"
    assert result.argv[result.argv.index("--infer-resource-profile") + 1] == "high"
    assert result.argv[result.argv.index("--infer-batch-size") + 1] == "12"
    assert result.argv[result.argv.index("--infer-workers") + 1] == "10"
    assert result.argv[result.argv.index("--infer-decode-threads") + 1] == "3"
    assert result.argv[result.argv.index("--infer-prefetch") + 1] == "4"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (PipelineSpec(), "Enable at least one pipeline step."),
        (PipelineSpec(steps=PipelineSteps(train=True)), "Pick at least one training pair."),
        (
            PipelineSpec(steps=PipelineSteps(extra_test=True), model_dir="/model"),
            "Pick at least one test pair.",
        ),
        (
            PipelineSpec(
                steps=PipelineSteps(infer=True),
                input_selection=PipelineSelection(folder="/videos", stems=["clip"]),
            ),
            "Model load folder is required when not training.",
        ),
        (
            PipelineSpec(steps=PipelineSteps(infer=True), model_dir="/model"),
            "Inference input folder is required.",
        ),
        (
            PipelineSpec(
                steps=PipelineSteps(infer=True),
                model_dir="/model",
                input_selection=PipelineSelection(folder="/videos"),
            ),
            "Pick at least one inference video.",
        ),
    ],
)
def test_build_command_validation_errors(spec, message):
    with pytest.raises(PipelineSpecError, match=message):
        build_pipeline_command(spec)
