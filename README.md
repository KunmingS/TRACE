# V-TRACE — Video-based Temporal Recognition and Annotation of Continuous Ethograms of Animal Behavior

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
gives you its address — open `http://localhost:8765` in Chrome or Edge — and you
type the rest of the commands at the `vtrace >` prompt, without repeating the
program's name (`demo predict`, not `vtrace demo predict`). Tab completes,
ctrl-r searches history, `exit` leaves.

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
vtrace demo predict     # predict on a held-out video, write its predictions
vtrace demo download    # fetch the videos (train ~19 GB, test ~11 GB)
vtrace demo train       # prep + train on the 70 official training videos
```

V-TRACE does not re-host CalMS21. The annotation CSVs ship with the package; the
videos come straight from the
[official CalMS21 release](https://data.caltech.edu/records/s0vdx-0k302)
(doi:10.22002/D1.1991), one member at a time over HTTP range requests, so
nothing downloads the full 28 GB archive and an interrupted fetch resumes by
skipping what is already on disk. `--split train` / `--split test` limits what
is pulled, and `--from /path/to/task1_videos_mp4.zip` reads a copy you already
have.

`vtrace demo predict` pulls the single video it needs and runs in about a minute.
`vtrace demo train` is a real training run on the full benchmark, not a smoke
test: 337 iterations per epoch over ten epochs, roughly three hours on one
modern GPU (~23 GB of VRAM).

If you use the demo data, cite CalMS21: Sun et al., *The Multi-Agent Behavior
Dataset: Mouse Dyadic Social Interactions*, NeurIPS 2021 Datasets & Benchmarks.

`vtrace demo download` assembles an ordinary model directory in `~/.vtrace/demo`, so
afterwards the plain commands work against it too:

```bash
vtrace predict --model-dir ~/.vtrace/demo/model --input /my/video.mp4
```

```bash
# Check whether PyPI has a newer V-TRACE release
vtrace update

# Train from selected video/annotation pairs
vtrace train --model maev2 --work-dir /my/dataset --pairs video01.mp4=video01_final.csv video02.mp4=video02.csv

# Evaluate the training artifact, or evaluate on held-out video/annotation pairs
vtrace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS
vtrace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --work-dir /my/testset --pairs video03.mp4=video03.csv

# Predict on new videos and write annotation drafts
vtrace predict --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --input /path/to/video.mp4 --threshold 0.25

# Run prep -> train -> predict end to end
vtrace pipeline --train --infer --model maev2 \
  --work-dir /my/dataset --pairs video01.mp4=video01_final.csv \
  --input /path/to/new/videos
```

`--pairs` is explicit: each item is `VIDEO_PATH=CSV_PATH`. A source video can
have multiple annotation CSVs beside it, such as `video01_draft.csv`,
`video01_final.csv`, or a reviewed prediction exported from the annotator; each
training or evaluation pair chooses the annotation file to use for that video.
Relative paths are resolved against `--work-dir`, so
`video01.mp4=video01_final.csv` means both files are inside the work directory.
Training creates a self-contained `model_YYYYMMDD_HHMMSS/` folder under
`--work-dir`.

V-TRACE annotation CSVs are time-based:

```csv
labelId,timestamp,endTimestamp
grooming,12.430,18.970
rearing,42.100,45.650
```

Prediction writes **one file per video**, named after it and placed beside it:

```
my_video.mp4
my_video.predict.json
```

Re-running prediction replaces that file rather than accumulating copies — a
video has exactly one current prediction. The JSON holds the merged behaviour
bouts with their confidences, in the shape the V-TRACE annotator imports, so the
same folder opened in `vtrace app` shows the video with its predictions ready for
review. `--output DIR` writes the files somewhere else, keeping the names.

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

Example — score the CalMS21 detector on the 19 official test videos:

```bash
torchrun --nproc_per_node=1 tools/test.py configs/calms21_distill_vmaeB.py \
    --checkpoint calms21_vitB_distilled_best.pth
```

## License

Apache 2.0. See [LICENSE](LICENSE).
