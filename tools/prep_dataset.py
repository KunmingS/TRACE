"""Prepare a raw dataset (videos + CSVs) into training metadata + annotations.

Wraps vtrace.data_prep.prepare_dataset() as a standalone script
so the CLI can run it as a subprocess.

Writes prep_result.json to the current working directory on success.
"""
import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("work_dir", type=str, help="Directory containing video/CSV files")
    parser.add_argument("--subset", choices=("train", "validation"), default="train",
                        help="Which subset every prepared video belongs to. Nothing is "
                             "split here: training data and evaluation data are separate "
                             "corpora, each prepared on its own.")
    parser.add_argument("--proxy-resolution", type=int, default=144,
                        help="Short side of the downscaled decode proxy built next to each "
                             "source video. 0 disables proxies and decodes from the originals.")
    parser.add_argument("--proxy-aspect", action="store_true",
                        help="Preserve the source aspect ratio in the proxy (scale=-2:R) "
                             "instead of squashing to a square. Required for pipelines that "
                             "crop after an aspect-preserving resize.")
    parser.add_argument("--proxy-crf", type=int, default=23,
                        help="H.264 CRF quality for the decode proxy")
    parser.add_argument("--proxy-workers", type=int, default=None,
                        help="Parallel workers for proxy encoding")
    parser.add_argument("--pairs", dest="explicit_pairs", nargs="*", default=None,
                        help="Restrict prep to explicit video+CSV pairs. Each item "
                             "must be VIDEO_PATH=CSV_PATH. Relative paths are resolved "
                             "against work_dir.")
    parser.add_argument("--include-stems", dest="include_stems",
                        nargs="*", default=None,
                        help="Internal/API compatibility: restrict auto-discovery to "
                             "these frontend pair stems.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write dataset.json/classmap.txt. "
                             "Defaults to a new model_ timestamp directory under work_dir.")
    parser.add_argument("--output", type=str, default="prep_result.json",
                        help="Output JSON path for results")
    args = parser.parse_args()

    from vtrace.data_prep import prepare_dataset
    from vtrace.model_artifacts import create_model_dir
    from vtrace.proxy_geometry import ProxyGeometry, align_up

    model_dir = args.output_dir or create_model_dir(args.work_dir)
    proxy_geometry = (
        ProxyGeometry(align_up(args.proxy_resolution), not args.proxy_aspect)
        if args.proxy_resolution > 0
        else None
    )
    model_dir, json_path, classmap_path = prepare_dataset(
        args.work_dir,
        subset=args.subset,
        proxy_geometry=proxy_geometry,
        proxy_crf=args.proxy_crf,
        proxy_workers=args.proxy_workers,
        included_stems=args.include_stems,
        explicit_pairs=args.explicit_pairs,
        output_dir=model_dir,
    )

    result = {
        "model_dir": model_dir,
        "dataset_json": json_path,
        "classmap_path": classmap_path,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    # The path is machine-facing — the caller passed it in and reads it back.
    # `prepare_dataset` has already said where the run folder is.


if __name__ == "__main__":
    main()
