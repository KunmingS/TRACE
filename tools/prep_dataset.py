"""Prepare a raw dataset (videos + CSVs) into clips + annotations.

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
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    parser.add_argument("--clip-frames", type=int, default=768, help="Frames per clip")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--output", type=str, default="prep_result.json",
                        help="Output JSON path for results")
    args = parser.parse_args()

    from trace_tad.data_prep import prepare_dataset

    clips_dir, json_path, classmap_path = prepare_dataset(
        args.dataset_path,
        clip_frames=args.clip_frames,
        train_ratio=args.train_ratio,
    )

    result = {
        "clips_dir": clips_dir,
        "json_path": json_path,
        "classmap_path": classmap_path,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Prep result saved to: {args.output}")


if __name__ == "__main__":
    main()
