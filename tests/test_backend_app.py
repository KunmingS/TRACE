from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_list_dirs_returns_children_for_existing_directory_without_trailing_slash(tmp_path):
    parent = tmp_path / "datasets"
    alpha = parent / "alpha"
    beta = parent / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir()

    response = client.get("/api/dirs", params={"prefix": str(parent)})

    assert response.status_code == 200
    returned = response.json()["dirs"]
    assert str(alpha) in returned
    assert str(beta) in returned


def test_list_paths_returns_matching_directories_and_filtered_files(tmp_path):
    root = tmp_path / "videos"
    root.mkdir()
    matching_dir = root / "video_folder"
    matching_dir.mkdir()
    mp4 = root / "video_clip.mp4"
    avi = root / "video_alt.avi"
    txt = root / "video_notes.txt"
    mp4.write_bytes(b"mp4")
    avi.write_bytes(b"avi")
    txt.write_text("ignore me")

    response = client.get(
        "/api/paths",
        params={
            "prefix": str(root / "video"),
            "extensions": ".mp4,.avi",
        },
    )

    assert response.status_code == 200
    returned = response.json()["paths"]
    assert {"path": str(matching_dir), "type": "dir"} in returned
    assert {"path": str(mp4), "type": "file"} in returned
    assert {"path": str(avi), "type": "file"} in returned
    assert {"path": str(txt), "type": "file"} not in returned

