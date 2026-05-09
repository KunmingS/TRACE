"""Prepare a raw dataset (videos + CSVs) into training metadata + annotations.

Wraps trace_tad.data_prep.prepare_dataset() as a standalone script
so it can be submitted as a background job via the job queue.

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
    parser.add_argument("--clip-frames", type=int, default=768, help="Frames per clip")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--reencode-clips", action="store_true",
                        help="Physically extract each clip with ffmpeg (CRF-18 re-encode). "
                             "Default is virtual clips: dataset.json records source_video + "
                             "frame offsets, no clip files written, zero quality loss.")
    parser.add_argument("--cache-mode", choices=["virtual", "cached_video"], default="virtual",
                        help="Dataset cache mode. 'cached_video' writes resized annotated "
                             "windows to model_dir/cache/videos and trains from those clips.")
    parser.add_argument("--cache-resolution", type=int, default=144,
                        help="Square resolution for cached_video clips")
    parser.add_argument("--cache-crf", type=int, default=23,
                        help="H.264 CRF quality for cached_video clips")
    parser.add_argument("--cache-workers", type=int, default=None,
                        help="Parallel workers for cached_video clip writing")
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

    from trace_tad.data_prep import prepare_dataset
    from trace_tad.model_artifacts import create_model_dir

    model_dir = args.output_dir or create_model_dir(args.work_dir)
    cache_mode = "physical" if args.reencode_clips else args.cache_mode
    model_dir, json_path, classmap_path = prepare_dataset(
        args.work_dir,
        clip_frames=args.clip_frames,
        train_ratio=args.train_ratio,
        virtual_clips=cache_mode == "virtual",
        cache_mode=cache_mode,
        cache_resolution=args.cache_resolution,
        cache_crf=args.cache_crf,
        cache_workers=args.cache_workers,
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
    print(f"Prep result saved to: {args.output}")


if __name__ == "__main__":
    main()
