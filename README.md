# V-TRACE — Video-based Temporal Recognition and Annotation of Continuous Ethograms of Animal Behavior

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22119863.svg)](https://doi.org/10.5281/zenodo.22119863)

V-TRACE turns untrimmed animal-behavior video into continuous ethograms: label a few
recordings in the browser, train a detector on them, and have it annotate the rest.

## Documentation

[kunmings.github.io/TRACE](https://kunmings.github.io/TRACE/)

## Install

Create a Python 3.9+ environment first. Choose one:

```bash
conda create -n vtrace python=3.11 pip
conda activate vtrace
```

```bash
mamba create -n vtrace python=3.11 pip
mamba activate vtrace
```

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
```

Then install V-TRACE and download the model weights:

```bash
python -m pip install vtrace-behavior
vtrace prepare --weights all
```

Training, evaluation and prediction need a CUDA-capable PyTorch environment.
Annotation does not — the browser app works anywhere.

## Usage

Run it with no arguments:

```bash
vtrace
```

That opens an interactive session. The annotator starts with it, so the screen
gives you its address — open `http://localhost:8765` in Chrome or Edge — and a
bordered **command box** opens under the start screen. The box is where the
command the annotator's configuration page writes goes: paste it, press Enter,
and it runs. Ordinary commands (`demo predict`, `help`) run from the same box,
without repeating the program's name. Esc closes the box and leaves the plain
`vtrace >` prompt, where `run` opens it again and `exit` leaves.

The folders in a pasted command are named, not located — `<train#8b1d0c47>` is
the folder called `train` whose contents hash to those digits — because a
browser never learns where a picked folder lives. The session looks each name
up on the machine it runs on and prints the path it found. From a plain shell
or a job script the same command is refused with a list of what to replace:
paste the page's shell form there and fill the paths in.

The annotator is a single static page. It reads videos and writes annotations
**on the computer running the browser**, through the File System Access API:
nothing is uploaded and there is no server-side state. Use it to open
recordings, define behaviors, mark events on the timeline, and review model
predictions; training, evaluation and prediction run from the command line.

`vtrace app` serves the annotator on its own, without the session — for scripts,
or when that is all you want.

Every command below also works directly from a shell, prefixed with `vtrace`.

## Try the demo

A walkthrough on CalMS21 mouse social behavior, using the *complete* Task-1
benchmark in its official split — 70 training videos (4.7 h, 2287 labelled
bouts) and 19 test videos (2.4 h, 1521 bouts):

```bash
vtrace demo predict     # predict on the 19 held-out test videos, write their predictions
vtrace demo download    # fetch the videos (train ~19 GB, test ~11 GB)
vtrace demo train       # prep + train on the 70 official training videos, then score the test split
```

V-TRACE does not re-host CalMS21. The annotation CSVs ship with the package; the
videos come straight from the
[official CalMS21 release](https://data.caltech.edu/records/s0vdx-0k302)
(doi:10.22002/D1.1991), one member at a time over HTTP range requests, so
nothing downloads the full 28 GB archive and an interrupted fetch resumes by
skipping what is already on disk. `--split train` / `--split test` limits what
is pulled, and `--from /path/to/task1_videos_mp4.zip` reads a copy you already
have.

`vtrace demo predict` labels all 19 test videos with the released checkpoint,
fetching them first if they are not on disk (~11 GB). It writes a `.pred.csv`
beside each video, plus a small `.pred.scores.npz` of per-frame class scores
from which it draws the precision-recall curve of each behavior over the whole
split and reports the frame mAP they integrate to. `--video STEM` runs just one
video; `--model-dir DIR` uses a run you trained instead of the released one.

`vtrace demo train` is a real training run on the full benchmark, not a smoke
test: 324 iterations per epoch over ten epochs, about two hours on one modern
GPU (~23 GB of VRAM). It cuts 15 of the 70 training videos out to pick the best
epoch, and when training is done it labels the 19 test videos with the finished
checkpoint exactly as `vtrace demo predict --model-dir` would — the same
per-video table, precision-recall curves and frame mAP — so a fresh run and the
released model are read the same way. Those 19 videos never took part in
training or epoch selection, so that mAP is the run's benchmark figure.

If you use the demo data, cite CalMS21: Sun et al., *The Multi-Agent Behavior
Dataset: Mouse Dyadic Social Interactions*, NeurIPS 2021 Datasets & Benchmarks.

Videos already on disk are checked against a shipped manifest of sizes and
CRC-32s rather than fetched again; `--verify` reads them through to catch a file
that is the right size but corrupt.

`vtrace demo download` assembles an ordinary model directory in `~/.vtrace/demo`,
so afterwards the plain commands work against it too:

```bash
vtrace predict --model-dir ~/.vtrace/demo/model --input /my/video.mp4
```

## Your own data

```bash
# Train from selected video/annotation pairs. Every epoch is checkpointed;
# with no evaluation videos, none of them is called best. The annotation is
# a CSV, or a <video>.json written by an earlier version of the annotator.
vtrace train --model maev2b --output /my/runs \
  --pairs /my/dataset/video01.mp4=/my/dataset/video01_final.csv \
          /my/dataset/video02.mp4=/my/dataset/video02.json

# Name evaluation videos and the training loop scores each epoch against
# them, which is what writes best.pth
vtrace train --model maev2b --output /my/runs \
  --pairs /my/dataset/video01.mp4=/my/dataset/video01_final.csv \
  --eval-pairs /my/testset/video03.mp4=/my/testset/video03.csv

# Score a model that already exists
vtrace eval --model-dir /my/runs/model_YYYYMMDD_HHMMSS \
  --pairs /my/testset/video03.mp4=/my/testset/video03.csv

# Predict on new videos
vtrace predict --model-dir /my/runs/model_YYYYMMDD_HHMMSS \
  --input /path/to/video.mp4 --threshold 0.25

# Chain the steps with `then`. A later step reuses the model the training
# step produced, so its timestamped folder never has to be typed. At the
# `vtrace >` prompt this can be typed over several lines, no backslashes.
vtrace train --model maev2b --output /my/runs \
    --pairs /my/dataset/video01.mp4=/my/dataset/video01_final.csv \
  then eval --pairs /my/testset/video03.mp4=/my/testset/video03.csv \
  then predict --input /path/to/new/videos

# Check PyPI for a newer release, and install it
vtrace update
```

Three model presets ship with the package:

| `--model` | Backbone | Training |
|---|---|---|
| `maev2b` (default) | VideoMAE V2 ViT-B/16 | ViT frozen, per-block adapters trained |
| `maev2b-distilled` | the same ViT-B | as above, but the adapters start from a V-JEPA 2 distillation instead of random init |
| `vjepa2` | V-JEPA 2 ViT-L | upper half of the encoder fine-tuned — heavier in every direction |

None of them names a dataset: prep supplies the paths and the class count per
run, so the same preset fits any corpus.

`--pairs` is explicit: each item is `VIDEO_PATH=CSV_PATH`. A source video can
have multiple annotation CSVs beside it, such as `video01_draft.csv`,
`video01_final.csv`, or a reviewed prediction exported from the annotator; each
training or evaluation pair chooses the annotation file to use for that video.
A pair names its own files, so there is no folder to set first; relative paths
resolve against the working directory. Where the run itself goes is a separate
question and is asked separately: `--output` is required, and the run writes a
self-contained `model_YYYYMMDD_HHMMSS/` folder there. Evaluation videos are
named the same way with `--eval-pairs`, and they are a corpus of their own —
nothing is held back from the training videos.

That choice decides what the run publishes. With evaluation videos, the loop
scores every epoch against them and the folder holds **`best.pth`**, the epoch
that scored highest. Without them nothing measured any epoch, so the folder
holds **`last.pth`** instead and every epoch's checkpoint stays under
`checkpoint/`. `vtrace eval` and `vtrace predict` read whichever is there, and
`vtrace eval --model-dir DIR` with no `--pairs` re-scores the run on the same
videos it validated against.

V-TRACE annotation CSVs are time-based:

```csv
labelId,timestamp,endTimestamp
grooming,12.430,18.970
rearing,42.100,45.650
```

Prediction writes **one file per video**, named after it and placed beside it:

```
my_video.mp4
my_video.pred.csv
```

Re-running prediction replaces that file rather than accumulating copies — a
video has exactly one current prediction. The file is an annotation CSV like the
one above, with a `score` column: the merged behaviour bouts with their
confidences, in the shape the V-TRACE annotator imports, so the same folder
opened in `vtrace app` shows the video with its predictions ready for review. A
`# trace-meta:` header line records the class list and the thresholds used.
`--output DIR` writes the files somewhere else, keeping the names.

## Shipped research configs and checkpoints

The repo carries the CalMS21 configurations behind the paper's results, and the
matching checkpoints are hosted on the GitHub `weights` release. **No checkpoint
is bundled in the repo or the wheel** — any config or `--checkpoint` argument
that names a registered file downloads it on first use (SHA256-verified, cached
under `~/.vtrace/pretrained`, override with `TRACE_WEIGHTS_DIR`):

| Config | Checkpoint | What it is |
|---|---|---|
| `configs/calms21_distill_vmaeB.py` | `calms21_vitB_distilled_best.pth` | The CalMS21 detector: a VideoMAE V2 ViT-B student, frozen ViT plus trained per-block adapters and a dense per-frame head. This is what `vtrace demo predict` runs. |
| `configs/calms21_vjepa2.py` | `calms21_vjepa2_teacher95.pth` | The V-JEPA2 ViT-L **encoder** fine-tuned on CalMS21 — the teacher the student above was distilled from. Encoder weights only, so it initialises a backbone rather than predicting on its own. |

Two pretrained bases complete the distillation chain: `vitB_videomaev2_k400.pth`
(the student's frozen ViT-B) and `vjepa2_vitl_encoder.pth` (the teacher's
starting encoder, before CalMS21 fine-tuning). To reproduce the distillation
itself, see `tools/distill_jepa_to_vmae.py` and `tools/bake_distill_adapters.py`.

These two carry the recipe, not the data: fill in the four dataset paths at the
top of the file (or override them at launch) to point at your CalMS21 copy —
`vtrace demo download` fetches exactly those videos. Then score the detector on
the 19 official test videos:

```bash
python tools/test.py configs/calms21_distill_vmaeB.py \
    --checkpoint calms21_vitB_distilled_best.pth \
    --cfg-options ann=... vid_train=... vid_test=... class_map=...
```

## License

Apache 2.0. See [LICENSE](LICENSE).
