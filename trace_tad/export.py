"""Export detection results to various formats (CSV, etc.)."""
import csv
import json
import os


def export_results(result_json, output_dir, fmt="csv", filename=None):
    """Convert result_detection.json to the requested format.

    Args:
        result_json: Path to result_detection.json or a dict already loaded.
        output_dir: Directory to write the exported file.
        fmt: Export format ("csv").
        filename: Optional output filename (without extension).

    Returns:
        Path to the exported file.
    """
    if isinstance(result_json, str):
        with open(result_json) as f:
            data = json.load(f)
    else:
        data = result_json

    results = data.get("results", {})
    os.makedirs(output_dir, exist_ok=True)

    if fmt == "csv":
        return _export_csv(results, output_dir, filename)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")


def _export_csv(results, output_dir, filename=None):
    """Write detections as CSV: video_name, start, end, label, score."""
    out_path = os.path.join(output_dir, (filename or "detections") + ".csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "start", "end", "label", "score"])
        for video_name, detections in sorted(results.items()):
            for det in detections:
                seg = det["segment"]
                writer.writerow([
                    video_name,
                    f"{seg[0]:.4f}",
                    f"{seg[1]:.4f}",
                    det["label"],
                    f"{det['score']:.6f}",
                ])
    return out_path
