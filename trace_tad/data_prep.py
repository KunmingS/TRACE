"""Dataset auto-preparation: scan for videos+CSVs, clip, and generate annotations.

Given a dataset directory containing raw videos and per-video CSV annotations,
this module:
1. Checks if clips/ subdirectory already exists with dataset.json (skip if so)
2. Extracts class names from CSVs → classmap.txt
3. Clips videos into fixed-length segments
4. Generates dataset.json in TRACE annotation format
"""

import csv
import json
import os
from pathlib import Path

import shutil
import subprocess

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _find_video_csv_pairs(dataset_path):
    """Find all (video_path, csv_path) pairs in a directory."""
    pairs = []
    for fname in sorted(os.listdir(dataset_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            video_path = os.path.join(dataset_path, fname)
            csv_path = os.path.join(dataset_path, os.path.splitext(fname)[0] + ".csv")
            if os.path.isfile(csv_path):
                pairs.append((video_path, csv_path))
    return pairs


def _extract_classes_from_csvs(csv_paths):
    """Collect all unique labels from CSV files, return sorted list."""
    labels = set()
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["labelId"].strip()
                if label:
                    labels.add(label)
    return sorted(labels)


def _process_video(video_path, csv_path, output_dir, clip_frames=768):
    """Process a single video: map CSV times to frames, extract clips.

    Returns dict of {clip_idx: {duration, frame, annotations}} for clips
    that contain at least one annotation.
    """
    video_name = Path(video_path).stem
    print(f"  Processing video: {video_name}")

    # Load CSV annotations
    annotations = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append({
                "labelId": row["labelId"].strip(),
                "timestamp": float(row["timestamp"]),
                "endTimestamp": float(row["endTimestamp"]),
            })
    print(f"    {len(annotations)} annotations")

    # Open video and get properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

    # Build timestamp array by reading all frames
    # Note: CAP_PROP_POS_MSEC after read() returns the *next* frame's position,
    # so we capture it *before* each read() to get the current frame's timestamp.
    print(f"    Building timestamp map...")
    timestamps = []
    while True:
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ret, _ = cap.read()
        if not ret:
            break
        timestamps.append(ts)
    ts_array = np.array(timestamps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Map annotation times to frame indices
    for anno in annotations:
        anno["start_frame"] = int(np.searchsorted(ts_array, anno["timestamp"], side="left"))
        anno["end_frame"] = int(np.searchsorted(ts_array, anno["endTimestamp"], side="right") - 1)

    # Convert annotations to clip-relative format
    num_clips = (total_frames + clip_frames - 1) // clip_frames
    clips_data = {}

    for clip_idx in range(num_clips):
        clip_start = clip_idx * clip_frames
        clip_end = min((clip_idx + 1) * clip_frames - 1, total_frames - 1)

        clip_annos = []
        for anno in annotations:
            if anno["end_frame"] < clip_start or anno["start_frame"] > clip_end:
                continue
            rel_start = max(0, anno["start_frame"] - clip_start)
            rel_end = min(clip_frames - 1, anno["end_frame"] - clip_start)
            if rel_start <= rel_end:
                clip_annos.append({
                    "frame_segment": [rel_start, rel_end],
                    "segment": [rel_start / fps, rel_end / fps],
                    "label": anno["labelId"],
                })

        if clip_annos:
            clips_data[clip_idx] = {
                "frame": min(clip_frames, total_frames - clip_start),
                "annotations": clip_annos,
            }

    clips_with_annos = sorted(clips_data.keys())
    print(f"    {len(clips_with_annos)} clips with annotations")

    if not clips_with_annos:
        cap.release()
        return clips_data

    # Extract clip videos using ffmpeg (avoids OpenCV h264_v4l2m2m issues)
    os.makedirs(output_dir, exist_ok=True)
    cap.release()

    use_ffmpeg = shutil.which("ffmpeg") is not None

    for clip_idx in clips_with_annos:
        clip_name = f"{video_name}_clip_{clip_idx}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        start_frame = clip_idx * clip_frames
        actual_frames = min(clip_frames, total_frames - start_frame)
        start_time = start_frame / fps
        duration_sec = actual_frames / fps

        if use_ffmpeg:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.4f}",
                "-i", video_path,
                "-t", f"{duration_sec:.4f}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-an",
                "-loglevel", "error",
                clip_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    WARNING: ffmpeg failed for {clip_name}: {result.stderr.strip()}")
                continue
        else:
            # Fallback to OpenCV (mp4v codec)
            cap2 = cv2.VideoCapture(video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"    WARNING: Could not open VideoWriter for {clip_name}, skipping")
                cap2.release()
                continue
            for _ in range(actual_frames):
                ret, frame = cap2.read()
                if not ret:
                    break
                out.write(frame)
            out.release()
            cap2.release()

        print(f"    Created clip {clip_idx}: {clip_name} ({actual_frames} frames)")

    # Set duration based on actual fps
    for clip_idx in clips_data:
        clips_data[clip_idx]["duration"] = clips_data[clip_idx]["frame"] / fps

    return clips_data


def _generate_classmap_from_json(json_path, classmap_path):
    """Extract sorted unique labels from a dataset JSON and write classmap.txt."""
    with open(json_path, "r", encoding="utf-8") as f:
        database = json.load(f)["database"]

    labels = set()
    for video_info in database.values():
        for anno in video_info.get("annotations", []):
            label = anno.get("label", "").strip()
            if label:
                labels.add(label)

    class_map = sorted(labels)
    with open(classmap_path, "w", encoding="utf-8") as f:
        for name in class_map:
            f.write(name + "\n")
    return class_map


def prepare_dataset(dataset_path, clip_frames=768, train_ratio=0.8):
    """Prepare a dataset directory for training.

    Scans for video+CSV pairs, clips videos, generates dataset.json and classmap.txt.
    Skips clipping if clips/ already exists with dataset.json.

    Args:
        dataset_path: Directory containing videos and CSV annotations.
        clip_frames: Number of frames per clip (default: 768).
        train_ratio: Fraction of clips for training (default: 0.8).

    Returns:
        (clips_dir, json_path, classmap_path) tuple of paths.
    """
    dataset_path = os.path.abspath(dataset_path)
    clips_dir = os.path.join(dataset_path, "clips")
    json_path = os.path.join(clips_dir, "dataset.json")
    classmap_path = os.path.join(clips_dir, "classmap.txt")

    # Check if already prepared (clips/ with dataset.json)
    if os.path.isdir(clips_dir) and os.path.isfile(json_path):
        # Ensure classmap exists — regenerate from JSON if missing
        if not os.path.isfile(classmap_path):
            _generate_classmap_from_json(json_path, classmap_path)
            print(f"Generated classmap: {classmap_path}")
        print(f"Dataset already prepared at {clips_dir}")
        return clips_dir, json_path, classmap_path

    # Find video-CSV pairs
    pairs = _find_video_csv_pairs(dataset_path)
    if not pairs:
        raise FileNotFoundError(
            f"No video+CSV pairs found in {dataset_path}. "
            "Expected matching .mp4 and .csv files (e.g., video1.mp4 + video1.csv)."
        )
    print(f"Found {len(pairs)} video-CSV pairs")

    # Extract class map from all CSVs
    csv_paths = [csv_path for _, csv_path in pairs]
    class_map = _extract_classes_from_csvs(csv_paths)
    if not class_map:
        raise ValueError(f"No labels found in CSV files in {dataset_path}")
    print(f"Classes: {class_map}")

    # Process each video
    all_clips_data = {}
    for video_path, csv_path in pairs:
        video_name = Path(video_path).stem
        clips_data = _process_video(video_path, csv_path, clips_dir, clip_frames)
        all_clips_data[video_name] = clips_data

    # Generate dataset.json with train/val split
    np.random.seed(42)
    database = {}
    for video_name, clips_data in all_clips_data.items():
        clip_indices = list(clips_data.keys())
        np.random.shuffle(clip_indices)
        train_count = int(len(clip_indices) * train_ratio)

        for i, clip_idx in enumerate(clip_indices):
            subset = "train" if i < train_count else "validation"
            clip_key = f"{video_name}_clip_{clip_idx}"
            database[clip_key] = {
                "duration": clips_data[clip_idx]["duration"],
                "frame": clips_data[clip_idx]["frame"],
                "subset": subset,
                "annotations": clips_data[clip_idx]["annotations"],
            }

    os.makedirs(clips_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"database": database}, f, indent=2, ensure_ascii=False)
    print(f"Dataset JSON saved: {json_path}")

    # Write classmap
    with open(classmap_path, "w", encoding="utf-8") as f:
        for name in class_map:
            f.write(name + "\n")
    print(f"Class map saved: {classmap_path}")

    return clips_dir, json_path, classmap_path
