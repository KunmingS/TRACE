"""Benchmark train dataloader resource profiles for a prepared TRACE dataset."""
import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Tune training dataloader resources")
    parser.add_argument("config", type=str, help="Training config path")
    parser.add_argument("--model-dir", required=True, help="Prepared model/dataset directory")
    parser.add_argument("--annotation-path", required=True, help="Prepared dataset.json")
    parser.add_argument("--class-map", required=True, help="Prepared classmap.txt")
    parser.add_argument("--output", default="train_tune_result.json", help="Output JSON path")
    parser.add_argument("--profiles-json", default=None, help="Optional JSON list of profiles")
    parser.add_argument("--max-batches", type=int, default=8, help="Batches to sample per profile")
    args = parser.parse_args()

    profiles = json.loads(args.profiles_json) if args.profiles_json else None

    from trace_tad.utils.train_tune import tune_train_resources

    result = tune_train_resources(
        args.config,
        model_dir=args.model_dir,
        annotation_path=args.annotation_path,
        class_map=args.class_map,
        profiles=profiles,
        max_batches=args.max_batches,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Train tune result saved to: {args.output}")


if __name__ == "__main__":
    main()
