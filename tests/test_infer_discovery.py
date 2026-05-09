import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_discover_videos_collapses_browser_ready_copies(tmp_path):
    from infer import discover_videos

    source = tmp_path / "trial.mkv"
    remux = tmp_path / "trial.mkv.remux.mp4"
    h264 = tmp_path / "trial.mkv.h264.mp4"
    other = tmp_path / "other.mp4"
    for path in (source, remux, h264, other):
        path.write_bytes(b"")

    videos = {Path(path).name for path in discover_videos(str(tmp_path))}

    assert videos == {"trial.mkv", "other.mp4"}


def test_discover_videos_keeps_browser_ready_copy_when_source_absent(tmp_path):
    from infer import discover_videos

    remux = tmp_path / "trial.mkv.remux.mp4"
    remux.write_bytes(b"")

    assert discover_videos(str(tmp_path)) == [str(remux)]


def test_annotated_video_lookup_uses_canonical_source_name():
    from trace_tad.video_annotation import _prediction_name_for_video

    assert _prediction_name_for_video("/data/trial.mkv.remux.mp4") == "trial"
    assert _prediction_name_for_video("/data/trial.mkv.h264.mp4") == "trial"
    assert _prediction_name_for_video("/data/trial.mkv") == "trial"


def test_prediction_csvs_are_named_like_source_video_and_copied_adjacent(tmp_path):
    from infer import write_prediction_csvs

    video = tmp_path / "trial.mp4"
    video.write_bytes(b"video")
    output_dir = tmp_path / "predict"
    predictions = {
        "trial": [
            {"label": "walk", "segment": [1.0, 2.25], "score": 0.9},
        ],
    }

    prediction_csvs, adjacent_csvs = write_prediction_csvs(
        [str(video)],
        predictions,
        str(output_dir),
    )

    expected = "labelId,timestamp,endTimestamp\nwalk,1.000,2.250\n"
    assert prediction_csvs["trial"] == str(output_dir / "trial.csv")
    assert adjacent_csvs["trial"] == str(tmp_path / "trial.csv")
    assert (output_dir / "trial.csv").read_text() == expected
    assert (tmp_path / "trial.csv").read_text() == expected


def test_prediction_csv_copy_does_not_overwrite_existing_annotation(tmp_path):
    from infer import write_prediction_csvs

    video = tmp_path / "trial.mp4"
    video.write_bytes(b"video")
    manual_csv = tmp_path / "trial.csv"
    manual_csv.write_text("labelId,timestamp,endTimestamp\nmanual,0.000,1.000\n")
    output_dir = tmp_path / "predict"
    predictions = {
        "trial": [
            {"label": "predicted", "segment": [3.0, 4.0], "score": 0.8},
        ],
    }

    _, adjacent_csvs = write_prediction_csvs(
        [str(video)],
        predictions,
        str(output_dir),
    )

    assert manual_csv.read_text() == "labelId,timestamp,endTimestamp\nmanual,0.000,1.000\n"
    assert adjacent_csvs["trial"] == str(tmp_path / "trial_predicted.csv")
    assert (tmp_path / "trial_predicted.csv").read_text() == (
        "labelId,timestamp,endTimestamp\npredicted,3.000,4.000\n"
    )


def test_prediction_csv_export_avoids_in_place_workdir_overwrite(tmp_path):
    from infer import write_prediction_csvs

    video = tmp_path / "trial.mp4"
    video.write_bytes(b"video")
    manual_csv = tmp_path / "trial.csv"
    manual_csv.write_text("labelId,timestamp,endTimestamp\nmanual,0.000,1.000\n")
    predictions = {
        "trial": [
            {"label": "predicted", "segment": [5.0, 6.0], "score": 0.7},
        ],
    }

    prediction_csvs, adjacent_csvs = write_prediction_csvs(
        [str(video)],
        predictions,
        str(tmp_path),
    )

    assert manual_csv.read_text() == "labelId,timestamp,endTimestamp\nmanual,0.000,1.000\n"
    assert prediction_csvs["trial"] == str(tmp_path / "trial_predicted.csv")
    assert adjacent_csvs["trial"] == str(tmp_path / "trial_predicted.csv")
    assert (tmp_path / "trial_predicted.csv").read_text() == (
        "labelId,timestamp,endTimestamp\npredicted,5.000,6.000\n"
    )
