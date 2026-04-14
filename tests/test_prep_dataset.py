import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prep_dataset_script_writes_result_for_prepared_dev_dataset(tmp_path):
    dataset = ROOT / "data" / "dev_test"
    output = tmp_path / "prep_result.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/prep_dataset.py",
            str(dataset),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Prep result saved to" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert Path(payload["clips_dir"]).resolve() == (dataset / "clips").resolve()
    assert Path(payload["json_path"]).resolve() == (dataset / "clips" / "dataset.json").resolve()
    assert Path(payload["classmap_path"]).resolve() == (dataset / "clips" / "classmap.txt").resolve()

