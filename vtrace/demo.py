"""The CalMS21 walkthrough behind `vtrace demo`.

Three steps a new user can run in order:

    vtrace demo download    # fetch the videos from CalMS21's own archive
    vtrace demo predict     # predict on a held-out video, write an annotation CSV
    vtrace demo train       # prep + train on the 70 official training videos

TRACE re-hosts none of CalMS21. The annotation CSVs ship with the package (they
are small, and they are the part in TRACE's own format); the videos come from the
dataset's record at CaltechDATA, which stays the single citable source.

That archive is one 28 GB ZIP, and pulling all of it to run a demo would be
absurd — so the videos are taken out of it a member at a time over HTTP range
requests (`vtrace.remote_zip`), which also means an interrupted download
resumes by simply skipping the files already on disk.

`download` then assembles a normal TRACE model directory out of the videos and
the released checkpoint, so every later step is a plain CLI command against
`--model-dir` and nothing about the demo is special-cased downstream.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zlib
from functools import lru_cache
from pathlib import Path

from vtrace import weights

# ── Where the videos come from ───────────────────────────────────────────────
# CalMS21, Sun et al. 2021 — Caltech Mouse Social Interactions.
# Open access, no login and no click-through. `SOURCE_SIZE` is required: a ZIP is
# read back-to-front, so the reader has to know where the end is.
SOURCE_NAME = "task1_videos_mp4.zip"
SOURCE_RECORD = "https://data.caltech.edu/records/s0vdx-0k302"
SOURCE_DOI = "10.22002/D1.1991"
SOURCE_URL = (
    "https://data.caltech.edu/api/records/s0vdx-0k302/files/"
    "task1_videos_mp4.zip/content"
)
SOURCE_SIZE = 28286565405
SOURCE_MD5 = "790b2ff6054c0889c3b0112b3a12eacc"
# Members are laid out as task1_videos_mp4/{train,test}/<stem>.mp4 — the same
# split TRACE uses, so a stem maps to a member by name alone.
SOURCE_PREFIX = "task1_videos_mp4"
SOURCE_CITATION = (
    "Sun et al., The Multi-Agent Behavior Dataset: Mouse Dyadic Social "
    "Interactions (CalMS21), NeurIPS 2021 Datasets & Benchmarks."
)

# The released CalMS21 model, pulled through the weight registry.
DEMO_CHECKPOINT = "calms21_vitB_distilled_best.pth"
DEMO_CONFIG = "configs/calms21_demo.py"

# Annotation CSVs, class map and the video manifest, shipped inside the
# package (~450 KB — no video).
PACKAGE_DATA = "demo_data/calms21"
MANIFEST_FILE = "videos.json"
SPLITS = ("train", "test")

# A source checkout can also carry the videos already; working from it skips the
# download entirely. Ignored by git, so it never reaches GitHub.
LOCAL_DATA_DIR = "data/calms21_demo"


def _checkout_copy() -> Path | None:
    """The demo data carried inside a source checkout, if it is there."""
    candidate = Path(__file__).resolve().parent.parent / LOCAL_DATA_DIR
    return candidate if candidate.is_dir() else None


def demo_dir() -> Path:
    """Where the demo lives.

    `TRACE_DEMO_DIR` wins, then a checkout that already carries the data, then
    the download cache under `~/.vtrace/demo`.
    """
    root = os.environ.get("TRACE_DEMO_DIR")
    if root:
        return Path(root)
    local = _checkout_copy()
    return local if local else weights._user_dir("demo")


def _package_config_path() -> str:
    """Absolute path to the demo config shipped with the package."""
    pkg_root = Path(__file__).resolve().parent.parent
    candidate = pkg_root / DEMO_CONFIG
    if candidate.is_file():
        return str(candidate)
    cwd_candidate = Path.cwd() / DEMO_CONFIG
    if cwd_candidate.is_file():
        return str(cwd_candidate)
    raise FileNotFoundError(
        f"Cannot find {DEMO_CONFIG}; run from a TRACE checkout or reinstall the package."
    )


def _pairs_in(folder: Path) -> list[str]:
    """`VIDEO=CSV` for every video in `folder` that has a CSV beside it."""
    pairs = []
    for video in sorted(folder.glob("*.mp4")):
        csv = video.with_suffix(".csv")
        if csv.is_file():
            pairs.append(f"{video}={csv}")
    return pairs


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink `src` to `dst`, copying if the platform refuses to link."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _package_data() -> Path:
    """The annotation CSVs, class map and video manifest shipped with the package."""
    return Path(__file__).resolve().parent / PACKAGE_DATA


@lru_cache(maxsize=1)
def manifest() -> dict:
    """Expected size/crc32/sha256 for every demo video, keyed by stem.

    The size and crc32 are the values the official archive's own index carries,
    so a file can be checked against CalMS21 itself without downloading anything
    and without trusting a checksum TRACE invented.
    """
    with open(_package_data() / MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)["videos"]


def wanted_stems(split: str) -> list[str]:
    """Video stems the demo needs for `split`."""
    return sorted(stem for stem, meta in manifest().items() if meta["split"] == split)


def _crc32(path: Path) -> str:
    crc = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            crc = zlib.crc32(chunk, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def check_video(path: Path, stem: str, deep: bool = False) -> str:
    """Classify one video: "ok", "missing", or a reason it is not usable.

    The default check is a stat: a video that is present at exactly the expected
    byte count is taken as good. That is what catches the case this exists for —
    a download interrupted partway — and it costs nothing, so `download` can run
    over a full 28 GB tree and decide it has nothing to do in milliseconds.
    `deep` additionally reads the bytes and checks the archive's CRC-32, which
    catches corruption a size alone cannot.
    """
    expected = manifest().get(stem)
    if expected is None:
        return "not part of the demo"
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != expected["size"]:
        return f"wrong size ({size} bytes, expected {expected['size']})"
    if deep and _crc32(path) != expected["crc32"]:
        return "failed CRC-32 check"
    return "ok"


def _install_annotations(root: Path) -> None:
    """Lay the shipped CSVs and class map out beside where the videos will go."""
    for split in SPLITS:
        target = root / "videos" / split
        target.mkdir(parents=True, exist_ok=True)
        for csv in (_package_data() / split).glob("*.csv"):
            destination = target / csv.name
            if not destination.is_file():
                shutil.copy2(csv, destination)
    classmap = root / "classmap.txt"
    if not classmap.is_file():
        shutil.copy2(_package_data() / "classmap.txt", classmap)


def missing_videos(root: Path, split: str, deep: bool = False) -> list[str]:
    """Stems of `split` that are absent or unusable, and so need fetching."""
    folder = root / "videos" / split
    return [stem for stem in wanted_stems(split)
            if check_video(folder / f"{stem}.mp4", stem, deep) != "ok"]


def _open_source(local_zip=None) -> zipfile.ZipFile:
    """The CalMS21 video archive — a local copy if given, else read remotely."""
    if local_zip:
        import zipfile

        return zipfile.ZipFile(local_zip)
    from vtrace.remote_zip import open_remote

    return open_remote(SOURCE_URL, SOURCE_SIZE)


def fetch_videos(root: Path, split: str, stems, local_zip=None) -> int:
    """Copy `stems` of `split` out of the CalMS21 archive into the demo tree.

    Each video is written to a temporary name and moved into place only once it
    is complete, so an interrupt never leaves a half-file that the next run would
    mistake for a finished download.
    """
    stems = list(stems)
    if not stems:
        return 0
    folder = root / "videos" / split
    folder.mkdir(parents=True, exist_ok=True)

    print(f"Reading the CalMS21 archive index ({SOURCE_NAME}) ...")
    with _open_source(local_zip) as archive:
        members = {}
        for info in archive.infolist():
            name = info.filename
            if name.endswith(".mp4") and f"/{split}/" in name:
                members[Path(name).stem] = info

        unknown = [stem for stem in stems if stem not in members]
        if unknown:
            print(f"Not in the archive: {', '.join(unknown[:5])}", file=sys.stderr)
            return 1

        total = sum(members[stem].file_size for stem in stems)
        plural = "s" if len(stems) > 1 else ""
        print(f"Fetching {len(stems)} {split} video{plural} "
              f"({total / 1e9:.2f} GB) from {SOURCE_RECORD}")
        for index, stem in enumerate(stems, 1):
            info = members[stem]
            destination = folder / f"{stem}.mp4"
            partial = destination.with_suffix(".mp4.part")
            print(f"  [{index}/{len(stems)}] {stem}.mp4  "
                  f"{info.file_size / 1e6:.0f} MB", flush=True)
            with archive.open(info) as source, open(partial, "wb") as out:
                shutil.copyfileobj(source, out, length=1 << 22)
            partial.replace(destination)
    return 0


def download(args=None) -> int:
    """Fetch the demo videos and assemble a ready-to-use model directory."""
    root = demo_dir()
    root.mkdir(parents=True, exist_ok=True)
    _install_annotations(root)

    # `--from` reads a copy of the archive already on disk, for offline installs
    # and for anyone who would rather download the ZIP once by hand.
    local_zip = getattr(args, "source", None)
    if local_zip:
        local_zip = Path(local_zip).expanduser()
        if not local_zip.is_file():
            print(f"No such file: {local_zip}", file=sys.stderr)
            return 1

    requested = getattr(args, "split", None) or "all"
    deep = bool(getattr(args, "verify", False))
    splits = SPLITS if requested == "all" else (requested,)

    if deep:
        print("Checking the videos already on disk against CalMS21's own CRC-32 "
              "values (this reads every file) ...")

    for split in splits:
        folder = root / "videos" / split
        stems = wanted_stems(split)
        needed, damaged = [], []
        for stem in stems:
            verdict = check_video(folder / f"{stem}.mp4", stem, deep)
            if verdict == "ok":
                continue
            needed.append(stem)
            if verdict != "missing":
                damaged.append(f"{stem}.mp4: {verdict}")

        if damaged:
            print(f"Re-fetching {len(damaged)} damaged {split} "
                  f"video{'s' if len(damaged) > 1 else ''}:")
            for line in damaged:
                print(f"    {line}")

        if not needed:
            print(f"All {len(stems)} {split} videos already present and correct "
                  f"in {folder} — nothing to download.")
            continue
        if len(needed) < len(stems):
            print(f"{len(stems) - len(needed)} of {len(stems)} {split} videos "
                  f"already on disk; fetching the remaining {len(needed)}.")

        code = fetch_videos(root, split, needed, local_zip)
        if code:
            return code

    return _assemble_model_dir(root)


def _assemble_model_dir(root: Path) -> int:
    """Build the demo's model directory out of the released checkpoint + class map.

    A normal TRACE model directory, so every later step is a plain CLI command
    against `--model-dir` and nothing about the demo is special-cased downstream.
    """
    videos = root / "videos"
    if not videos.is_dir():
        print(f"Demo bundle looks incomplete: {videos} is missing.", file=sys.stderr)
        return 1

    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)
    try:
        checkpoint = Path(weights.resolve(DEMO_CHECKPOINT))
    except RuntimeError as exc:
        # The registry knows the file but no host served it. A traceback here
        # tells the reader nothing they can act on.
        print(
            f"{exc}\n\n"
            f"The demo needs {DEMO_CHECKPOINT}. If you have it already, put it in\n"
            f"    {weights.cache_dir()}\n"
            f"or point TRACE_WEIGHTS_DIR at the directory holding it.",
            file=sys.stderr,
        )
        return 1
    _link_or_copy(checkpoint, model_dir / "best.pth")

    classmap = root / "classmap.txt"
    if not classmap.is_file():
        print(f"Demo bundle is missing {classmap}.", file=sys.stderr)
        return 1
    shutil.copy2(classmap, model_dir / "classmap.txt")
    # Freeze the architecture the released checkpoint was trained with, the same
    # way a real training run does, so a later edit of the shipped config cannot
    # change how this model is rebuilt.
    from vtrace.config import Config
    from vtrace.model_artifacts import RESOLVED_CONFIG_NAME

    cfg = Config.fromfile(_package_config_path())
    cfg.merge_from_dict({"model.num_classes": len(
        (root / "classmap.txt").read_text(encoding="utf-8").split()
    )})
    cfg.dump(model_dir / RESOLVED_CONFIG_NAME)
    (model_dir / "config.txt").write_text(_package_config_path() + "\n", encoding="utf-8")

    lines = [f"\nDemo ready in {root}"]
    for split in SPLITS:
        have = len(wanted_stems(split)) - len(missing_videos(root, split))
        note = "training videos" if split == "train" else "held-out videos"
        lines.append(f"  videos/{split:<6}{have}/{len(wanted_stems(split))} {note}")
    lines.append(f"  model/       {checkpoint.name}")
    lines.append("\nNext:  vtrace demo predict     (or:  vtrace demo train)")
    print("\n".join(lines))
    return 0


def ensure_videos(root: Path, split: str, stems=None) -> None:
    """Make sure `stems` of `split` are on disk, fetching whatever is not.

    The check is a stat per file, so calling this before every step costs
    nothing when the data is already there — which is the normal case after the
    first run. Only what is actually absent gets fetched.
    """
    stems = list(stems) if stems is not None else wanted_stems(split)
    folder = root / "videos" / split
    needed = [stem for stem in stems
              if check_video(folder / f"{stem}.mp4", stem) != "ok"]
    if not needed:
        return

    have = len(stems) - len(needed)
    if have:
        print(f"{have} of the {len(stems)} {split} videos are already on disk.")
    print("Interrupt any time — finished videos are kept, and the next run "
          "carries on from there.\n")
    if fetch_videos(root, split, needed):
        raise SystemExit(f"Could not fetch the {split} videos.")


def _require_ready() -> Path:
    """The demo root, with its model directory built if it is not there yet."""
    root = demo_dir()
    if not (root / "model" / "config.txt").is_file():
        _install_annotations(root)
        if _assemble_model_dir(root):
            raise SystemExit("Could not set up the demo model directory.")
        print()
    return root


def predict(args=None) -> int:
    """Predict on one held-out demo video and write an annotation CSV."""
    from vtrace.cli import _require_cuda
    from vtrace.steps import InferRequest, run_infer

    _require_cuda()
    root = _require_ready()
    # The shortest held-out video, so the first thing a new user runs finishes
    # quickly — CalMS21 test videos range from under 3 minutes to over 13. Chosen
    # from the manifest rather than from what is on disk, so the demo predicts on
    # the same video whether or not the rest has been downloaded.
    stem = min(wanted_stems("test"), key=lambda name: manifest()[name]["size"])
    ensure_videos(root, "test", [stem])
    video = root / "videos" / "test" / f"{stem}.mp4"
    model_dir = root / "model"

    print(
        "Running:\n"
        f"    vtrace predict --model-dir {model_dir} \\\n"
        f"        --input {video} --threshold 0.25\n"
    )
    result = run_infer(InferRequest(
        model_dir=str(model_dir),
        input=str(video),
        threshold=0.25,
    ))
    if result.ok:
        print(
            f"\nWrote {video.with_suffix('').name}.predict.json next to the video, in\n"
            f"    {video.parent}\n"
            f"Open that folder in `vtrace app` to review the predictions."
        )
    else:
        print(f"\nPrediction failed (exit code {result.returncode}); see {result.log_file}")
    return result.returncode


def train(args=None) -> int:
    """Prep the demo clips and fine-tune on them for a couple of epochs."""
    import json

    from vtrace.cli import _require_cuda
    from vtrace.model_artifacts import create_model_dir
    from vtrace.steps import PrepRequest, TrainRequest, run_prep, run_train

    _require_cuda()
    root = _require_ready()
    train_dir = root / "videos" / "train"
    ensure_videos(root, "train")
    pairs = _pairs_in(train_dir)
    if not pairs:
        raise SystemExit(f"No video/CSV pairs found in {train_dir}.")

    config_path = _package_config_path()
    print(
        "Running the equivalent of:\n"
        f"    vtrace train --config {config_path} \\\n"
        f"        --work-dir {train_dir} \\\n"
        f"        --pairs {' '.join(Path(p.split('=')[0]).name + '=' + Path(p.split('=')[1]).name for p in pairs)}\n"
    )

    # Derive the decode-proxy geometry from the config the run will actually use,
    # exactly as `vtrace train` does. Left to its default, prep would build 144px
    # proxies while this config decodes at 224 and train on upsampled frames.
    from vtrace.config import Config
    from vtrace.proxy_geometry import config_geometry

    geometry = config_geometry(Config.fromfile(config_path))

    model_dir = create_model_dir(str(train_dir))
    prep = run_prep(PrepRequest(
        work_dir=str(train_dir),
        model_dir=model_dir,
        proxy_resolution=geometry.short_side,
        proxy_aspect=not geometry.square,
        explicit_pairs=pairs,
    ))
    if not prep.ok:
        print(f"\nDataset prep failed (exit code {prep.returncode}); see {prep.log_file}")
        return prep.returncode

    with open(Path(prep.work_dir) / "prep_result.json", encoding="utf-8") as f:
        prep_result = json.load(f)

    result = run_train(TrainRequest(
        config_path=config_path,
        model_dir=prep_result["model_dir"],
        dataset_dir=prep_result["model_dir"],
        annotation_path=prep_result["dataset_json"],
        class_map=prep_result["classmap_path"],
    ))
    if result.ok:
        print(f"\nTraining finished. Model directory: {prep_result['model_dir']}")
    else:
        print(f"\nTraining failed (exit code {result.returncode}); see {result.log_file}")
    return result.returncode
