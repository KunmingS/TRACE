"""The CalMS21 walkthrough behind `vtrace demo`.

Three steps a new user can run in order:

    vtrace demo download    # fetch the videos from CalMS21's own archive
    vtrace demo predict     # predict on a held-out video, write an annotation CSV
    vtrace demo train       # prep + train on the 70 official training videos

TRACE re-hosts none of CalMS21. The annotation CSVs ship with the package (they
are small, and they are the part in TRACE's own format); the videos come from the
the official CalMS21 release, which stays the single citable source.

That archive is one 28 GB ZIP, and pulling all of it to run a demo would be
absurd — so the videos are taken out of it a member at a time over HTTP range
requests (`vtrace.remote_zip`), which also means an interrupted download picks
up by skipping the files already on disk. Only whole videos survive an
interrupt: the members are DEFLATE-compressed, so there is no restart point
inside one and the video in flight is fetched again from its first byte.

`download` then assembles a normal TRACE model directory out of the videos and
the released checkpoint, so every later step is a plain CLI command against
`--model-dir` and nothing about the demo is special-cased downstream.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import zlib
from functools import lru_cache
from pathlib import Path

from vtrace import weights

# ── Where the videos come from ───────────────────────────────────────────────
# The official CalMS21 release (Caltech Mouse Social Interactions, Sun et al.
# 2021), archived in Caltech's data repository. Open access, no login and no
# click-through. `SOURCE_SIZE` is required: a ZIP is
# read back-to-front, so the reader has to know where the end is.
SOURCE_NAME = "task1_videos_mp4.zip"
SOURCE_RECORD = "https://data.caltech.edu/records/s0vdx-0k302"
SOURCE_DOI = "10.22002/D1.1991"
SOURCE_URL = (
    f"https://data.caltech.edu/api/records/s0vdx-0k302/files/"
    f"{SOURCE_NAME}/content"
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


# The start screen's accent, so the demo reads as the same tool.
ACCENT = "cyan"

# Four transfers at once. The archive is one big ZIP on a public host, so the
# limit that matters is politeness rather than local bandwidth: four is enough
# to keep a fast link busy on a set of ~300 MB members without behaving like a
# scraper. `--jobs` moves it either way.
DEFAULT_JOBS = 4


def _console():
    """A Rich console for stdout, or None when Rich is not installed.

    Shared with the start screen, which is also where NO_COLOR, width and
    not-a-terminal handling come from — so a redirected `demo download > log`
    writes plain text without any of this having to know.
    """
    from vtrace import splash

    return splash.get_console()


class _PlainReport:
    """The progress reporter for a terminal that cannot draw one.

    Same interface as the Rich one, so `fetch_videos` has a single code path:
    one line per file, exactly what the command printed before bars existed.
    """

    class _File:
        def advance(self, amount):
            pass

        def done(self):
            pass

    def __init__(self, count):
        self.count = count
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin(self, index, stem, size):
        with self._lock:
            print(f"  [{index}/{self.count}] {stem}.mp4  {size / 1e6:.0f} MB",
                  flush=True)
        return self._File()


class _RichReport:
    """One bar for the whole split, plus one per download in flight.

    The overall bar is what a reader actually wants from a 19 GB download — how
    much is left and how long that will take. The per-file bars are there so a
    stalled connection reads as a stall rather than as an overall bar that
    happens to be moving slowly.
    """

    class _File:
        def __init__(self, report, task):
            self.report = report
            self.task = task

        def advance(self, amount):
            self.report.progress.advance(self.task, amount)
            self.report.progress.advance(self.report.overall, amount)

        def done(self):
            self.report.progress.remove_task(self.task)

    def __init__(self, progress, total_bytes, count):
        self.progress = progress
        self.total_bytes = total_bytes
        self.count = count
        self.overall = None

    def __enter__(self):
        self.progress.start()
        self.overall = self.progress.add_task(
            f"[bold]all {self.count}[/]", total=self.total_bytes)
        return self

    def __exit__(self, *exc):
        self.progress.stop()
        return False

    def begin(self, index, stem, size):
        # add_task/advance/remove_task are all guarded by Progress's own lock,
        # so worker threads can call these directly.
        task = self.progress.add_task(
            f"[{ACCENT}]{index}/{self.count}[/] {stem}.mp4", total=size)
        return self._File(self, task)


def _progress(console, total_bytes, count):
    """A progress reporter for `count` files totalling `total_bytes`."""
    # A bar that cannot animate is worse than the lines it replaced: redirected
    # to a file it renders once, at 0%, and says nothing for the rest of the run.
    if console is None or not console.is_terminal:
        return _PlainReport(count)
    try:
        from rich.progress import (BarColumn, DownloadColumn, Progress,
                                   TaskProgressColumn, TextColumn,
                                   TimeRemainingColumn, TransferSpeedColumn)
    except ImportError:
        return _PlainReport(count)
    progress = Progress(
        TextColumn("  {task.description}"),
        BarColumn(bar_width=None, complete_style=ACCENT, finished_style="green"),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(compact=True),
        console=console,
        transient=False,
    )
    return _RichReport(progress, total_bytes, count)


def _partial_owner(name: str):
    """The pid that owns a scratch file, or None if the name predates them.

    Scratch files are `<stem>.mp4.<pid>.<tid>.part`. Anything else is from an
    older version of this command and belongs to nobody.
    """
    parts = name.split(".")
    if len(parts) < 4 or parts[-1] != "part":
        return None
    try:
        return int(parts[-3])
    except ValueError:
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Running, just not ours to signal.
        return True
    except OSError:
        return True
    return True


def _sweep_orphan_partials(folder: Path) -> None:
    """Delete scratch files no live download owns.

    A hard kill leaves one behind per transfer in flight, and none of them can
    ever be resumed — the members are deflate streams. Skipping the ones whose
    owner is still running is what keeps this safe to call while a second
    `demo download` is working in the same folder.
    """
    for leftover in folder.glob("*.part"):
        owner = _partial_owner(leftover.name)
        if owner is not None and owner != os.getpid() and _process_alive(owner):
            continue
        if owner == os.getpid():
            continue
        try:
            leftover.unlink()
        except OSError:
            pass


def fetch_videos(root: Path, split: str, stems, local_zip=None, jobs=1) -> int:
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
    _sweep_orphan_partials(folder)

    console = _console()
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
        jobs = max(1, int(jobs or 1))
        # Where it lands and how to stop it. A tens-of-GB download that names
        # neither is asking the reader to either trust it or kill the terminal.
        if console:
            console.print(
                f"Fetching [bold]{len(stems)}[/] {split} video{plural} "
                f"([bold]{total / 1e9:.2f} GB[/]) from "
                f"[{ACCENT}]{SOURCE_RECORD}[/]", highlight=False)
            console.print(f"  into [{ACCENT}]{folder}[/]", highlight=False)
            console.print("  [yellow]ctrl-c[/] stops it.", highlight=False)
        else:
            print(f"Fetching {len(stems)} {split} video{plural} "
                  f"({total / 1e9:.2f} GB) from {SOURCE_RECORD}")
            print(f"  into {folder}")
            print("  ctrl-c stops it.", flush=True)

        # One archive handle per worker: a ZipFile keeps a single file position,
        # so sharing one across threads would have them seek out from under each
        # other. The extra cost is one small ranged GET of the ZIP tail per
        # worker for the whole run, not per file.
        local = threading.local()
        opened = []
        opened_lock = threading.Lock()

        def archive_for_thread():
            handle = getattr(local, "archive", None)
            if handle is None:
                handle = _open_source(local_zip)
                local.archive = handle
                with opened_lock:
                    opened.append(handle)
            return handle

        def fetch_one(index, stem):
            info = members[stem]
            destination = folder / f"{stem}.mp4"
            # The scratch name carries the process and thread, so two runs of
            # `demo download` over one demo folder — two terminals, or a second
            # run started before the first finished — cannot end up writing the
            # same scratch file and renaming it out from under each other.
            # Whoever finishes a given video last wins, with identical bytes.
            partial = folder / (f"{stem}.mp4.{os.getpid()}."
                                f"{threading.get_ident()}.part")
            tracker = report.begin(index, stem, info.file_size)
            try:
                mine = archive_for_thread()
                # By name, not by the ZipInfo read on another handle: let each
                # archive resolve the member through its own central directory.
                with mine.open(info.filename) as source, open(partial, "wb") as out:
                    # Not copyfileobj: the progress bar needs to see each chunk
                    # go past.
                    while True:
                        chunk = source.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
                        tracker.advance(len(chunk))
            except BaseException:
                # Nothing can ever be resumed from this file: the member is a
                # deflate stream, so the next attempt has to start from its
                # first byte regardless. Keeping it would only be most of a
                # gigabyte of disk that no run will read.
                partial.unlink(missing_ok=True)
                raise
            finally:
                tracker.done()
            partial.replace(destination)

        with _progress(console, total, len(stems)) as report:
            work = list(enumerate(stems, 1))
            if jobs == 1:
                for index, stem in work:
                    fetch_one(index, stem)
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    futures = [pool.submit(fetch_one, index, stem)
                               for index, stem in work]
                    try:
                        for future in futures:
                            future.result()
                    except BaseException:
                        # Ctrl-C, or one member failing. Stop handing out work;
                        # the running transfers clean up their own .part files
                        # as they unwind.
                        for future in futures:
                            future.cancel()
                        raise
        for handle in opened:
            try:
                handle.close()
            except OSError:
                pass
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
    jobs = int(getattr(args, "jobs", None) or DEFAULT_JOBS)
    splits = SPLITS if requested == "all" else (requested,)

    console = _console()

    def say(markup, plain):
        """One line, coloured where the terminal can take colour."""
        if console:
            console.print(markup, highlight=False)
        else:
            print(plain)

    if deep:
        say("Checking the videos already on disk against CalMS21's own "
            "[bold]CRC-32[/] values [dim](this reads every file)[/] ...",
            "Checking the videos already on disk against CalMS21's own CRC-32 "
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
            plural = "s" if len(damaged) > 1 else ""
            say(f"[yellow]Re-fetching {len(damaged)} damaged {split} "
                f"video{plural}:[/]",
                f"Re-fetching {len(damaged)} damaged {split} video{plural}:")
            for line in damaged:
                say(f"    [yellow]{line}[/]", f"    {line}")

        if not needed:
            say(f"[green]All {len(stems)} {split} videos already present and "
                f"correct[/] — nothing to download.",
                f"All {len(stems)} {split} videos already present and correct "
                f"— nothing to download.")
            continue
        if len(needed) < len(stems):
            say(f"[green]{len(stems) - len(needed)} of {len(stems)}[/] {split} "
                f"videos already on disk; fetching the remaining "
                f"[bold]{len(needed)}[/].",
                f"{len(stems) - len(needed)} of {len(stems)} {split} videos "
                f"already on disk; fetching the remaining {len(needed)}.")

        code = fetch_videos(root, split, needed, local_zip, jobs)
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

    _print_ready(root, checkpoint)
    return 0


# What each next step is for, in the same voice as the start screen's rows.
_NEXT_STEPS = (
    ("demo predict", "label the 19 held-out test videos, write predictions"),
    ("demo train", "train on the 70 official training videos, then score the test split"),
)


def _print_ready(root: Path, checkpoint: Path) -> None:
    """The closing summary: what is on disk, and what to type next.

    Laid out like the start screen — a bold command in a fixed column, a dim
    note beside it — so arriving here does not feel like arriving in a
    different program.
    """
    from vtrace import splash

    inventory = []
    for split in SPLITS:
        total = len(wanted_stems(split))
        have = total - len(missing_videos(root, split))
        # Named for what they are in CalMS21: training draws from the train
        # split (and cuts its own validation out of it); the test split is the
        # benchmark's, and nothing in training touches it.
        note = ("train split" if split == "train"
                else "test split, held out")
        inventory.append((f"videos/{split}", have, total, note))

    console = _console()
    if console is None:
        print(f"\nDemo ready in {root}")
        for name, have, total, note in inventory:
            print(f"  {name:<14}{have}/{total} {note}")
        print(f"  {'model/':<14}{checkpoint.name}")
        print("\nNext")
        for command, note in _NEXT_STEPS:
            print(f"  {splash.typed('vtrace ' + command):<14}{note}")
        return

    from rich.table import Table
    from rich.text import Text

    # One shared label column across both grids, wide enough for the longest
    # entry in either — a fixed width truncates `vtrace demo predict`, and
    # measuring each grid on its own would step the two out of line.
    labels = [f"  {name}" for name, *_ in inventory] + ["  model/"] + [
        f"  {splash.typed('vtrace ' + command)}" for command, _note in _NEXT_STEPS]
    label_width = max(len(label) for label in labels)

    def grid(rows):
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True, width=label_width)
        table.add_column(no_wrap=True, width=6, justify="right")
        table.add_column(style="dim")
        for row in rows:
            table.add_row(*row)
        return table

    console.print()
    console.print(f"[bold]Demo ready in[/] [{ACCENT}]{root}[/]", highlight=False)
    body = []
    for name, have, total, note in inventory:
        # Green only when the split is whole: a partial count is a fact about
        # the disk, not an achievement.
        style = "green" if have == total else "yellow"
        body.append((Text(f"  {name}", style="bold"),
                     Text(f"{have}/{total}", style=style), note))
    body.append((Text("  model/", style="bold"),
                 Text(""), checkpoint.name))
    console.print(grid(body))

    console.print()
    console.print("[bold]Next[/]", highlight=False)
    console.print(grid([(Text(f"  {splash.typed('vtrace ' + command)}", style="bold"),
                         Text(""), note) for command, note in _NEXT_STEPS]))
    console.print()


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
    if fetch_videos(root, split, needed, jobs=DEFAULT_JOBS):
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
    """Predict on the held-out demo videos and write an annotation CSV per video.

    Every test video by default: the 19 are the benchmark's held-out split, and
    a prediction over all of them is what a model should be judged on. `--video`
    narrows the run to one stem for a quick look, and `--model-dir` swaps the
    released checkpoint for a run `demo train` produced.
    """
    from vtrace.cli import _model_info_or_exit, _require_cuda

    _require_cuda()
    root = _require_ready()

    stems = wanted_stems("test")
    only = getattr(args, "video", None)
    if only:
        only = Path(only).name
        if only.endswith(".mp4"):
            only = only[:-len(".mp4")]
        if only not in stems:
            listed = "\n".join(f"  {stem}" for stem in stems)
            raise SystemExit(f"{only} is not one of the demo's test videos:\n{listed}")
        stems = [only]

    chosen = getattr(args, "model_dir", None)
    model_dir = Path(chosen).expanduser().resolve() if chosen else root / "model"
    _model_info_or_exit(str(model_dir))
    return _predict_test_split(root, model_dir, stems)


def _predict_test_split(root: Path, model_dir: Path, stems=None) -> int:
    """Label the held-out test videos with `model_dir` and report on the result.

    The one path both `demo predict` and the end of `demo train` go through, so
    a freshly trained run is reported exactly the way the released checkpoint
    is: the same per-video table, the same precision-recall curves, the same
    mAP. Prediction files land beside the test videos, where the annotator
    finds them.
    """
    from vtrace.steps import InferRequest, run_infer

    test_dir = root / "videos" / "test"
    stems = list(stems) if stems else wanted_stems("test")
    ensure_videos(root, "test", stems)
    videos = [test_dir / f"{stem}.mp4" for stem in stems]

    console = _console()
    what = (f"[{ACCENT}]{videos[0].name}[/]" if len(videos) == 1
            else f"[{ACCENT}]{len(videos)}[/] held-out test videos in [{ACCENT}]{test_dir}[/]")
    if console:
        console.print(f"Predicting on {what}", highlight=False)
        console.print(f"  with [{ACCENT}]{model_dir}[/]\n", highlight=False)
    else:
        plain = videos[0].name if len(videos) == 1 else f"{len(videos)} held-out test videos in {test_dir}"
        print(f"Predicting on {plain}\n  with {model_dir}\n")

    result = run_infer(InferRequest(
        model_dir=str(model_dir),
        input=str(videos[0]) if len(videos) == 1 else str(test_dir),
        included_stems=None if len(videos) == 1 else stems,
        # No floor: the sweep below needs the whole score range, and the
        # per-class cutoff it settles on is what the file is written at.
        threshold=0.0,
        tune_threshold=True,
        # Per-frame scores beside each CSV, for the precision-recall curves the
        # overview draws over the whole split.
        frame_scores=True,
    ))
    if not result.ok:
        print(f"\nPrediction failed (exit code {result.returncode}); "
              f"see {result.log_file}")
        return result.returncode
    if len(videos) == 1:
        _print_prediction_summary(videos[0].parent / f"{stems[0]}.pred.csv", videos[0])
    else:
        _print_predictions_overview(videos)
    return result.returncode


def _print_predictions_overview(videos) -> None:
    """One row per video: what was written and how it lines up with the annotation.

    The single-video summary draws ethograms and spans; nineteen of those would
    bury the answer. Here each video gets a line, and the table closes with the
    mean, so the run reads at a glance and any one video can be opened in the
    annotator for the full picture.
    """
    from vtrace import splash

    rows = []
    for video in videos:
        csv_path = video.parent / f"{video.stem}.pred.csv"
        if not csv_path.is_file():
            rows.append((video.name, None, None, None, None))
            continue
        bouts = _read_bouts(csv_path)
        duration = _video_duration(video)
        span = sum(end - start for _label, start, end, _score in bouts)
        share = span / duration if duration else None
        scored = _score_written(csv_path, video)
        macro = scored[1] if scored else None
        rows.append((video.name, len(bouts), duration, share, macro))
    macros = [macro for *_rest, macro in rows if macro is not None]
    mean_macro = sum(macros) / len(macros) if macros else None
    written = sum(1 for _name, bouts, *_rest in rows if bouts is not None)
    folder = videos[0].parent

    console = _console()
    if console is None:
        print(f"\n{written} prediction files written to {folder}")
        print(f"  {'video':<32}{'bouts':>6}{'length':>9}{'labelled':>10}{'macro F1':>10}")
        for name, bouts, duration, share, macro in rows:
            if bouts is None:
                print(f"  {name:<32}{'no prediction file':>35}")
                continue
            print(f"  {name:<32}{bouts:>6}{duration:>8.0f}s"
                  f"{(f'{share * 100:.0f}%' if share is not None else ''):>10}"
                  f"{(f'{macro:.2f}' if macro is not None else ''):>10}")
        if mean_macro is not None:
            print(f"  {'mean':<32}{'':>6}{'':>9}{'':>10}{mean_macro:>10.2f}")
            print("  macro F1 is frame-level, against each video's CalMS21 annotation, at "
                  "thresholds tuned on that same video — a demonstration, not the benchmark mAP")
        _print_pr_curves(None, videos)
        print("\nReview them in the annotator")
        print(f"  1. {splash.typed('vtrace app')}   (already running in this session)"
              if splash.IN_SESSION else f"  1. {splash.typed('vtrace app')}")
        print(f"  2. Open Video Folder → {folder}")
        print("  3. Pick any video; its predictions load with it.")
        return

    from rich.table import Table
    from rich.text import Text

    console.print()
    console.print(f"[bold]{written} prediction files[/] → [{ACCENT}]{folder}[/]",
                  highlight=False)
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True, width=32, style="bold")
    table.add_column(no_wrap=True, width=9, justify="right")
    table.add_column(no_wrap=True, width=8, justify="right", style="dim")
    table.add_column(no_wrap=True, width=9, justify="right", style="dim")
    table.add_column(no_wrap=True, width=9, justify="right")
    table.add_row(Text(""), Text("bouts", style="dim"), Text("length", style="dim"),
                  Text("labelled", style="dim"), Text("macro F1", style="dim"))
    for name, bouts, duration, share, macro in rows:
        if bouts is None:
            table.add_row(f"  {name}", Text("no prediction file", style="yellow"), "", "", "")
            continue
        table.add_row(
            f"  {name}", f"{bouts} bouts", f"{duration:.0f}s",
            f"{share * 100:.0f}%" if share is not None else "",
            Text(f"{macro:.2f}", style="green" if macro >= 0.7 else "yellow")
            if macro is not None else Text(""),
        )
    if mean_macro is not None:
        table.add_row(Text("  mean", style="bold"), "", "", "",
                      Text(f"{mean_macro:.2f}",
                           style="green" if mean_macro >= 0.7 else "yellow"))
    console.print(table)
    if mean_macro is not None:
        console.print("  [dim]macro F1 is frame-level, against each video's CalMS21 "
                      "annotation, at thresholds tuned on that same video — a "
                      "demonstration, not the benchmark mAP[/]", highlight=False)
    _print_pr_curves(console, videos)
    console.print()
    console.print("[bold]Review them in the annotator[/]", highlight=False)
    where = ("[dim](already running in this session)[/]" if splash.IN_SESSION
             else f"[bold]{splash.typed('vtrace app')}[/]")
    console.print(f"  1. {where}", highlight=False)
    console.print(f"  2. [bold]Open Video Folder[/] → [{ACCENT}]{folder}[/]", highlight=False)
    console.print("  3. Pick any video — its predictions load with it.", highlight=False)
    console.print()


def _pooled_pr_curves(videos):
    """Precision-recall per class over every frame of every video, and its AP.

    Pools frames across videos before ranking, which is how the benchmark mAP is
    computed: one curve per class over the whole split, not an average of
    per-video curves. AP is the sklearn-style area the evaluator reports, so the
    mean here is the number `vtrace eval` would print for the same files.

    Returns (curves, mean_ap, frames, videos_scored, absent) with `curves` a list
    of (label, recall, precision, ap, positives) and `absent` the classes with no
    annotated frame in these videos (no curve, and no part in the mean), or None
    when no video has both a score file and an annotation.
    """
    import numpy as np

    from vtrace.evaluations.precision import _average_precision_sklearn_style

    pooled_scores, pooled_truth = {}, {}
    frames = scored = 0
    for video in videos:
        scores_path = video.parent / f"{video.stem}.pred.scores.npz"
        truth_path = video.with_suffix(".csv")
        sidecar = Path(str(video) + ".pts.npy")
        if not (scores_path.is_file() and truth_path.is_file() and sidecar.is_file()):
            continue
        try:
            data = np.load(scores_path)
            pts = np.load(sidecar)
            scores = data["scores"].astype(np.float32)
            labels = [str(label) for label in data["labels"]]
        except Exception:
            continue
        if scores.shape[0] != len(pts) or scores.shape[1] != len(labels):
            continue
        truth = _frame_labels(_read_bouts(truth_path), pts)
        frames += len(pts)
        scored += 1
        for col, label in enumerate(labels):
            gt = truth.get(label)
            gt = np.zeros(len(pts), dtype=bool) if gt is None else gt
            pooled_scores.setdefault(label, []).append(scores[:, col])
            pooled_truth.setdefault(label, []).append(gt)
    if not pooled_scores:
        return None

    curves, absent = [], []
    for label, chunks in pooled_scores.items():
        score = np.concatenate(chunks)
        truth = np.concatenate(pooled_truth[label])
        positives = int(truth.sum())
        if positives == 0:
            absent.append(label)
            continue
        ap = _average_precision_sklearn_style(score, truth)
        order = np.argsort(score, kind="mergesort")[::-1]
        score, truth = score[order], truth[order]
        # One point per distinct score, as sklearn ranks ties together.
        cut = np.r_[np.where(np.diff(score))[0], score.size - 1]
        tps = np.cumsum(truth)[cut].astype(np.float64)
        fps = 1 + cut - tps
        precision = tps / np.maximum(tps + fps, 1.0)
        recall = tps / positives
        curves.append((label, recall, precision, ap, positives))
    if not curves:
        return None
    mean_ap = sum(curve[3] for curve in curves) / len(curves)
    return curves, mean_ap, frames, scored, absent


# Braille cells give a 2x4 dot grid per character: fine enough that three
# precision-recall curves read as curves rather than staircases in a 60-column
# plot. Bit for dot (column, row), rows top to bottom.
_BRAILLE_BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


def _plot_pr_curves(curves, width=60, height=12):
    """Rows of (character, curve index or None) drawing recall (x) vs precision (y).

    Each curve is sampled at every dot column: the precision at the first point
    whose recall reaches that column's recall, joined vertically to the previous
    column's so a cliff in precision draws as a wall, not a gap. Where curves
    cross, the later one owns the cell's colour; the dots of both survive.
    """
    import numpy as np

    columns, rows = width * 2, height * 4
    dots = [[0] * width for _ in range(height)]
    owner = [[None] * width for _ in range(height)]

    def mark(x, y, index):
        cell_x, cell_y = x // 2, y // 4
        dots[cell_y][cell_x] |= _BRAILLE_BITS[x % 2][y % 4]
        owner[cell_y][cell_x] = index

    for index, (_label, recall, precision, _ap, _positives) in enumerate(curves):
        previous = None
        for x in range(columns):
            target = x / (columns - 1)
            at = int(np.searchsorted(recall, target, side="left"))
            if at >= len(recall):
                break
            y = int(round((1.0 - float(precision[at])) * (rows - 1)))
            y = max(0, min(rows - 1, y))
            low, high = (y, y) if previous is None else (min(previous, y), max(previous, y))
            for step in range(low, high + 1):
                mark(x, step, index)
            previous = y
    return [[(chr(0x2800 + dots[r][c]), owner[r][c]) for c in range(width)]
            for r in range(height)]


def _print_pr_curves(console, videos) -> None:
    """The precision-recall curves over the whole split, and the mAP they integrate to.

    Drawn from the per-frame score files inference wrote beside the CSVs, against
    the CalMS21 annotation of each video. This is the benchmark quantity: frame
    mAP over all frames of every video, threshold-free, so it says how well the
    model ranks frames rather than how well one cutoff happened to land.
    """
    pooled = _pooled_pr_curves(videos)
    if pooled is None:
        return
    curves, mean_ap, frames, scored, absent = pooled
    width, height = 60, 12
    grid = _plot_pr_curves(curves, width, height)
    axis = {0: "1.0", height // 2: "0.5", height - 1: "0.0"}
    # Seven columns of margin (indent, precision label, axis) put `0` under the
    # first plotted column and `1` under the last.
    margin = " " * 7
    footer = f"{margin}0{'recall'.center(width - 2)}1"
    what = (f"frame-level over {scored} videos, {frames:,} frames, all frames scored — "
            f"the benchmark protocol")

    if console is None:
        glyphs = _ETHOGRAM_GLYPHS
        print(f"\nPrecision-recall over the split  mAP {mean_ap:.3f}")
        print(f"  {what}")
        for row, cells in enumerate(grid):
            line = "".join(glyphs[index % len(glyphs)] if index is not None else " "
                           for _char, index in cells)
            print(f"  {axis.get(row, ''):>3} │{line}")
        print(f"      └{'─' * width}")
        print(footer)
        for index, (label, _recall, _precision, ap, positives) in enumerate(curves):
            print(f"  {glyphs[index % len(glyphs)]} {label:<16}AP {ap:.3f}   "
                  f"{positives:,} positive frames")
        for label in absent:
            print(f"    {label:<16}no annotated frames in these videos — not in the mean")
        return

    from rich.text import Text

    console.print()
    console.print(f"[bold]Precision-recall over the split[/]  mAP [bold]{mean_ap:.3f}[/]",
                  highlight=False)
    console.print(f"  [dim]{what}[/]", highlight=False)
    colours = [_ETHOGRAM_COLOURS[index % len(_ETHOGRAM_COLOURS)] for index in range(len(curves))]
    for row, cells in enumerate(grid):
        # No base style on the line: a dim base would dim the curves too.
        line = Text()
        line.append(f"  {axis.get(row, ''):>3} │", style="dim")
        for char, index in cells:
            line.append(char, style=colours[index] if index is not None else None)
        console.print(line)
    console.print(Text(f"      └{'─' * width}", style="dim"))
    console.print(Text(footer, style="dim"))
    for index, (label, _recall, _precision, ap, positives) in enumerate(curves):
        line = Text("  ")
        line.append("━━", style=colours[index])
        line.append(f" {label:<16}", style="bold")
        line.append(f"AP {ap:.3f}", style=ACCENT)
        line.append(f"   {positives:,} positive frames", style="dim")
        console.print(line)
    for label in absent:
        console.print(f"     [bold]{label:<16}[/][dim]no annotated frames in these videos "
                      f"— not in the mean[/]", highlight=False)


def _video_duration(video: Path) -> float:
    """Seconds of `video`, from the PTS sidecar prep and inference both leave.

    0.0 when there is none — the summary then drops the "share of the
    recording" column rather than inventing a denominator.
    """
    sidecar = Path(str(video) + ".pts.npy")
    if not sidecar.is_file():
        return 0.0
    try:
        import numpy as np

        pts = np.load(sidecar)
        return float(pts[-1] - pts[0]) if len(pts) > 1 else 0.0
    except Exception:
        return 0.0


def _read_bouts(path: Path):
    """(label, start, end, score) for each row of a prediction CSV."""
    # The same reader prep uses, so a `# trace-meta:` header and any column
    # padding are skipped here exactly as they are there.
    from vtrace.data_prep import _csv_dict_reader

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in _csv_dict_reader(f):
            try:
                rows.append((
                    row["labelId"].strip(),
                    float(row["timestamp"]),
                    float(row["endTimestamp"]),
                    float(row.get("score") or 0.0),
                ))
            except (KeyError, ValueError):
                continue
    return rows


def _frame_labels(bouts, pts):
    """label -> boolean array over `pts`, True where that label is annotated."""
    import numpy as np

    origin = pts[0]
    masks = {}
    for label, start, end, _score in bouts:
        mask = masks.get(label)
        if mask is None:
            mask = masks[label] = np.zeros(len(pts), dtype=bool)
        first = int(np.searchsorted(pts, origin + start, side="left"))
        last = int(np.searchsorted(pts, origin + end, side="right"))
        mask[first:last] = True
    return masks


def _score_written(csv_path: Path, video: Path):
    """Score the prediction file as written against the CSV beside the video.

    Returns (rows, macro_f1, frames) or None when there is no ground truth —
    which is the normal case outside the demo. The cutoffs were chosen during
    inference, where the raw per-frame scores are; this only reports how the
    file that came out of it lines up.
    """
    import json as _json

    truth_path = video.with_suffix(".csv")
    sidecar = Path(str(video) + ".pts.npy")
    if not truth_path.is_file() or not sidecar.is_file():
        return None
    try:
        import numpy as np

        pts = np.load(sidecar)
    except Exception:
        return None
    if len(pts) < 2:
        return None

    # The per-class cutoffs inference settled on, recorded in the header.
    cutoffs = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "trace-meta:" in line:
                marker = line.index("trace-meta:") + len("trace-meta:")
                try:
                    chosen = _json.loads(line[marker:]).get("threshold")
                except ValueError:
                    chosen = None
                if isinstance(chosen, dict):
                    cutoffs = chosen

    import numpy as np

    truth = _frame_labels(_read_bouts(truth_path), pts)
    predicted = _frame_labels(_read_bouts(csv_path), pts)
    rows = []
    for label in sorted(set(truth) | set(predicted)):
        gt = truth.get(label)
        pr = predicted.get(label)
        gt = np.zeros(len(pts), dtype=bool) if gt is None else gt
        pr = np.zeros(len(pts), dtype=bool) if pr is None else pr
        tp = int((gt & pr).sum())
        fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((label, precision, recall, f1, float(cutoffs.get(label, 0.0))))
    if not rows:
        return None
    return rows, sum(row[3] for row in rows) / len(rows), len(pts)


# One colour per behaviour, in the order classes are first seen. The same
# family the annotator's timeline uses, so a strip printed here and the strip
# drawn in the browser read as the same picture.
_ETHOGRAM_COLOURS = ("#89b4fa", "#f9c74f", "#f38ba8", "#94e2d5", "#f5c2e7",
                     "#a6e3a1", "#fab387")
# Fallback glyphs for a terminal with no colour: colour is the class, so
# without it the class has to be the character.
_ETHOGRAM_GLYPHS = "#=*+xo~"


def _ethogram_columns(bouts, duration, width):
    """One label (or None) per column, sampling the middle of each column.

    Sampling rather than accumulating: a column is a moment, and a behaviour
    that covers less than one column of the recording should not widen to fill
    one. Overlaps resolve to whichever bout starts first, which is also how the
    annotator draws them.
    """
    columns = [None] * width
    if duration <= 0:
        return columns
    ordered = sorted(bouts, key=lambda b: (b[1], b[2]))
    for index in range(width):
        moment = (index + 0.5) * duration / width
        for label, start, end, *_rest in ordered:
            if start <= moment <= end:
                columns[index] = label
                break
    return columns


def _print_ethograms(console, truth_bouts, predicted_bouts, duration, labels):
    """Two strips, the annotation above the prediction, on one time axis.

    The numbers say how well they agree. This says *where* — which is the
    question anyone about to open the annotator actually has.
    """
    width = 76
    if console is not None:
        width = max(40, min(96, console.width - 16))
    truth = _ethogram_columns(truth_bouts, duration, width)
    predicted = _ethogram_columns(predicted_bouts, duration, width)

    # A ruler that lands its labels under the columns they name.
    ticks = [" "] * width
    marks = 4
    for step in range(marks + 1):
        position = min(width - 1, round(step * (width - 1) / marks))
        stamp = f"{duration * position / max(1, width - 1):.0f}s"
        for offset, ch in enumerate(stamp):
            at = position + offset if step < marks else position - len(stamp) + 1 + offset
            if 0 <= at < width:
                ticks[at] = ch
    ruler = "".join(ticks)

    if console is None:
        glyphs = {label: _ETHOGRAM_GLYPHS[i % len(_ETHOGRAM_GLYPHS)]
                  for i, label in enumerate(labels)}
        render = lambda cols: "".join(glyphs.get(c, ".") if c else "." for c in cols)
        print("\n  annotated   " + render(truth))
        print("  predicted   " + render(predicted))
        print("              " + ruler)
        print("  " + "   ".join(f"{glyphs[l]} {l}" for l in labels))
        return

    from rich.text import Text

    colours = {label: _ETHOGRAM_COLOURS[i % len(_ETHOGRAM_COLOURS)]
               for i, label in enumerate(labels)}

    def strip(columns):
        text = Text()
        for label in columns:
            if label is None:
                text.append("━", style="#3a4a46")
            else:
                text.append("█", style=colours[label])
        return text

    console.print()
    # Assemble rather than concatenate: `Text + Text` carries the left operand's
    # style across the whole result, which would paint every block bold.
    console.print(Text.assemble(("  annotated   ", "bold"), strip(truth)))
    console.print(Text.assemble(("  predicted   ", "bold"), strip(predicted)))
    console.print(Text.assemble("              ", (ruler, "dim")))
    legend = Text("  ")
    for label in labels:
        legend.append("█ ", style=colours[label])
        legend.append(f"{label}   ", style="dim")
    console.print(legend)


def _print_spans(console, truth_bouts, predicted_bouts, limit=8):
    """The bouts themselves, annotation beside prediction, earliest first."""
    def rows(bouts):
        return [f"{label} {start:.1f}–{end:.1f}s"
                for label, start, end, *_rest in sorted(bouts, key=lambda b: b[1])]

    left, right = rows(truth_bouts), rows(predicted_bouts)
    shown = max(len(left), len(right))
    extra = shown - limit
    left, right = left[:limit], right[:limit]
    width = max([len(r) for r in left] + [22]) + 4

    head_left = f"annotated ({len(truth_bouts)})"
    head_right = f"predicted ({len(predicted_bouts)})"
    if console is None:
        print(f"\n  {head_left:<{width}}{head_right}")
        for index in range(max(len(left), len(right))):
            a = left[index] if index < len(left) else ""
            b = right[index] if index < len(right) else ""
            print(f"  {a:<{width}}{b}")
        if extra > 0:
            print(f"  … {extra} more")
        return

    from rich.table import Table
    from rich.text import Text

    table = Table.grid(padding=(0, 4))
    table.add_column(no_wrap=True, width=width)
    table.add_column(no_wrap=True)
    table.add_row(Text(f"  {head_left}", style="bold"),
                  Text(head_right, style="bold"))
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else ""
        b = right[index] if index < len(right) else ""
        table.add_row(Text(f"  {a}", style="dim"), Text(b, style="dim"))
    console.print()
    console.print(table)
    if extra > 0:
        console.print(f"  [dim]… {extra} more[/]", highlight=False)


def _print_prediction_summary(csv_path: Path, video: Path) -> None:
    """What the model found, and where to go and look at it.

    The per-frame log the engine prints is a record of a computation. This is
    the answer: which behaviors, how much of the recording each one covers, and
    the one click that puts it on a timeline.
    """
    from vtrace import splash

    if not csv_path.is_file():
        print(f"\nNo prediction file at {csv_path}")
        return
    scored = _score_written(csv_path, video)
    bouts = _read_bouts(csv_path)
    span = sum(end - start for _label, start, end, _score in bouts)
    duration = _video_duration(video)

    by_label = {}
    for label, start, end, score in bouts:
        count, total, best = by_label.get(label, (0, 0.0, 0.0))
        by_label[label] = (count + 1, total + (end - start), max(best, score))
    ordered = sorted(by_label.items(), key=lambda kv: kv[1][1], reverse=True)

    console = _console()
    header = (f"{len(bouts)} bouts across {len(ordered)} "
              f"behavior{'s' if len(ordered) != 1 else ''}")
    if console is None:
        print(f"\n{header} → {csv_path.name}")
        for label, (count, total, best) in ordered:
            share = f"{total / duration * 100:4.1f}%" if duration else "     "
            print(f"  {label:<16}{count:>3} bouts{total:>8.1f}s {share}"
                  f"   best {best:.2f}")
        if duration:
            print(f"  {'—':<16}{'':>3}       {span:>8.1f}s "
                  f"{span / duration * 100:4.1f}% of {duration:.0f}s labelled")
        truth_bouts = _read_bouts(video.with_suffix(".csv"))
        if truth_bouts and duration:
            seen = []
            for label, *_ in sorted(truth_bouts + bouts, key=lambda b: b[1]):
                if label not in seen:
                    seen.append(label)
            _print_ethograms(None, truth_bouts, bouts, duration, seen)
            _print_spans(None, truth_bouts, bouts)
        if scored:
            rows, macro, frames = scored
            print(f"\nAgainst the CalMS21 annotation for this video "
                  f"({frames} frames, threshold tuned per class)")
            print(f"  {'':<16}{'threshold':>11}{'precision':>11}"
                  f"{'recall':>9}{'F1':>7}")
            for label, precision, recall, f1, cutoff in rows:
                print(f"  {label:<16}{cutoff:>11.2f}{precision:>11.2f}"
                      f"{recall:>9.2f}{f1:>7.2f}")
            print(f"  {'macro F1':<16}{'':>11}{'':>11}{'':>9}{macro:>7.2f}")
        print("\nReview them in the annotator")
        print(f"  1. {splash.typed('vtrace app')}   (already running in this session)"
              if splash.IN_SESSION else f"  1. {splash.typed('vtrace app')}")
        print(f"  2. Open Video Folder → {video.parent}")
        print(f"  3. Pick {video.name}; the predictions load with it.")
        return

    from rich.table import Table
    from rich.text import Text

    console.print()
    console.print(f"[bold]{header}[/] → [{ACCENT}]{csv_path.name}[/]", highlight=False)
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True, width=16, style="bold")
    table.add_column(no_wrap=True, width=9, justify="right")
    table.add_column(no_wrap=True, width=9, justify="right")
    table.add_column(no_wrap=True, width=6, justify="right", style="dim")
    table.add_column(style="dim")
    for label, (count, total, best) in ordered:
        share = f"{total / duration * 100:.1f}%" if duration else ""
        table.add_row(f"  {label}", f"{count} bouts",
                      Text(f"{total:.1f}s", style="green"), share,
                      f"best {best:.2f}")
    console.print(table)
    if duration:
        console.print(f"  [dim]{span:.1f}s of {duration:.0f}s labelled "
                      f"({span / duration * 100:.0f}% of the recording)[/]",
                      highlight=False)

    truth_bouts = _read_bouts(video.with_suffix(".csv"))
    if truth_bouts and duration:
        seen = []
        for label, *_ in sorted(truth_bouts + bouts, key=lambda b: b[1]):
            if label not in seen:
                seen.append(label)
        _print_ethograms(console, truth_bouts, bouts, duration, seen)
        _print_spans(console, truth_bouts, bouts)

    if scored:
        rows, macro, frames = scored
        console.print()
        console.print(
            f"[bold]Against the CalMS21 annotation for this video[/] "
            f"[dim]({frames} frames, threshold tuned per class)[/]", highlight=False)
        marks = Table.grid(padding=(0, 2))
        marks.add_column(no_wrap=True, width=16, style="bold")
        marks.add_column(no_wrap=True, width=10, justify="right")
        marks.add_column(no_wrap=True, width=10, justify="right")
        marks.add_column(no_wrap=True, width=8, justify="right")
        marks.add_column(no_wrap=True, width=6, justify="right")
        marks.add_row(Text(""), Text("threshold", style="dim"),
                      Text("precision", style="dim"),
                      Text("recall", style="dim"), Text("F1", style="dim"))
        for label, precision, recall, f1, cutoff in rows:
            marks.add_row(f"  {label}", Text(f"{cutoff:.2f}", style=ACCENT),
                          f"{precision:.2f}", f"{recall:.2f}",
                          Text(f"{f1:.2f}", style="green" if f1 >= 0.7 else "yellow"))
        marks.add_row(Text("  macro F1", style="bold"), Text(""), Text(""),
                      Text(""),
                      Text(f"{macro:.2f}",
                           style="green" if macro >= 0.7 else "yellow"))
        console.print(marks)
        console.print("  [dim]thresholds tuned on this video, so this is the "
                      "best it can do here — not a held-out estimate, and not "
                      "the benchmark mAP[/]", highlight=False)

    console.print()
    console.print("[bold]Review them in the annotator[/]", highlight=False)
    where = ("[dim](already running in this session)[/]" if splash.IN_SESSION
             else f"[bold]{splash.typed('vtrace app')}[/]")
    console.print(f"  1. {where}", highlight=False)
    console.print(f"  2. [bold]Open Video Folder[/] → [{ACCENT}]{video.parent}[/]",
                  highlight=False)
    console.print(f"  3. Pick [{ACCENT}]{video.name}[/] — the predictions load with it.",
                  highlight=False)
    console.print()


# One fifth of the training videos, held back to pick the best epoch. Small
# enough that the model still sees most of a small corpus, large enough that a
# rare behavior lands in it: at 0.2 of 70 videos, the 18 with any `attack` put
# three or four on the validation side.
VALIDATION_SHARE = 0.2


def _split_for_validation(pairs, share=VALIDATION_SHARE, seed=42):
    """Cut `pairs` into (train, validation), stratified by behavior.

    Seeded, so the demo splits the same way every time it is run and two people
    following the walkthrough are looking at the same numbers.
    """
    from vtrace.data_prep import stratified_split

    split = stratified_split(
        pairs, [("train", 1.0 - share), ("validation", share)], seed=seed)
    return split["train"], split["validation"]


def train(args=None) -> int:
    """Prep the demo videos, train on them, then score the run on the test split."""
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

    # A validation split is cut out of CalMS21's *train* split, not taken from
    # its test split. The 19 test videos are the benchmark's held-out set: score
    # each epoch on them and the checkpoint has been chosen by looking at the
    # answers, and `demo predict` would then be reporting on a video the run had
    # already been tuned against. They are labelled exactly once, after training,
    # through the same path `demo predict` takes — the number that run can be
    # judged on, shown the same way for a fresh run as for the released model.
    #
    # The cut is stratified on which behaviors each video contains, because they
    # are far from evenly spread: only 18 of the 70 train videos have any
    # `attack` at all, and a random fifth of them could easily contain none.
    train_pairs, eval_pairs = _split_for_validation(pairs)

    config_path = _package_config_path()
    console = _console()
    lines = [
        (f"[bold]Training on[/] [{ACCENT}]{len(train_pairs)}[/] CalMS21 videos",
         f"Training on {len(train_pairs)} CalMS21 videos"),
        (f"  [dim]{len(eval_pairs)} more, split out of the same {len(pairs)} by "
         f"behavior, score each epoch[/]",
         f"  {len(eval_pairs)} more, split out of the same {len(pairs)} by "
         f"behavior, score each epoch"),
        (f"  [dim]the benchmark's {len(wanted_stems('test'))}-video test split "
         f"is scored once, after training[/]",
         f"  the benchmark's {len(wanted_stems('test'))}-video test split "
         f"is scored once, after training"),
        (f"  [dim]config[/] [{ACCENT}]{Path(config_path).name}[/]\n",
         f"  config {Path(config_path).name}\n"),
    ]
    for markup, text in lines:
        if console:
            console.print(markup, highlight=False)
        else:
            print(text)

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
        explicit_pairs=train_pairs,
    ))
    if not prep.ok:
        print(f"\nDataset prep failed (exit code {prep.returncode}); see {prep.log_file}")
        return prep.returncode

    with open(Path(prep.work_dir) / "prep_result.json", encoding="utf-8") as f:
        prep_result = json.load(f)

    eval_annotation = eval_data_dir = None
    if eval_pairs:
        eval_prep = run_prep(PrepRequest(
            work_dir=str(train_dir),
            model_dir=str(Path(prep_result["model_dir"]) / "eval_data"),
            subset="validation",
            proxy_resolution=geometry.short_side,
            proxy_aspect=not geometry.square,
            explicit_pairs=eval_pairs,
        ))
        if not eval_prep.ok:
            print(f"\nEvaluation prep failed (exit code {eval_prep.returncode}); "
                  f"see {eval_prep.log_file}")
            return eval_prep.returncode
        with open(Path(eval_prep.work_dir) / "prep_result.json", encoding="utf-8") as f:
            eval_result = json.load(f)
        eval_annotation = eval_result["dataset_json"]
        eval_data_dir = eval_result["model_dir"]

    result = run_train(TrainRequest(
        config_path=config_path,
        model_dir=prep_result["model_dir"],
        dataset_dir=prep_result["model_dir"],
        annotation_path=prep_result["dataset_json"],
        class_map=prep_result["classmap_path"],
        eval_annotation_path=eval_annotation,
        eval_data_dir=eval_data_dir,
    ))
    if not result.ok:
        print(f"\nTraining failed (exit code {result.returncode}); "
              f"see {result.log_file}")
        return result.returncode
    model_dir = Path(prep_result["model_dir"])
    _print_trained(model_dir)
    code = _predict_test_split(root, model_dir)
    _print_use_it(model_dir)
    return code


def _print_trained(model_dir: Path) -> None:
    """Where the run landed, before its test-split prediction starts."""
    published = "best.pth" if (model_dir / "best.pth").is_file() else "last.pth"
    console = _console()
    if console is None:
        print(f"\nTraining finished → {model_dir}")
        print(f"  {published}   the checkpoint later steps load\n")
        return
    console.print()
    console.print(f"[bold]Training finished[/] → [{ACCENT}]{model_dir}[/]",
                  highlight=False)
    console.print(f"  [bold]{published}[/]   [dim]the checkpoint later steps load[/]",
                  highlight=False)
    console.print()


def _print_use_it(model_dir: Path) -> None:
    """The commands that use the run, after its test-split report."""
    from vtrace import splash

    predict = splash.typed(f"vtrace demo predict --model-dir {model_dir}")
    own = splash.typed(f"vtrace predict --model-dir {model_dir} --input <video>")
    console = _console()
    if console is None:
        print("Predict with it")
        print(f"  {predict}   the 19 test videos again")
        print(f"  {own}   a video of your own")
        return
    console.print("[bold]Predict with it[/]", highlight=False)
    console.print(f"  [{ACCENT}]{predict}[/]   [dim]the 19 test videos again[/]", highlight=False)
    console.print(f"  [{ACCENT}]{own}[/]   [dim]a video of your own[/]", highlight=False)
    console.print()
