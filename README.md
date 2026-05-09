# TRACE — Temporal Recognition of Animal Behaviors Captured from Video

TRACE is a temporal action detection system for animal behavior analysis in untrimmed video.

![TRACE Annotator](site/assets/screenshot.png)

## Documentation

[kunmings.github.io/TRACE](https://kunmings.github.io/TRACE/)

## Install

Create a Python 3.9+ environment first. Choose one:

```bash
conda create -n trace-tad python=3.11 pip
conda activate trace-tad
```

```bash
mamba create -n trace-tad python=3.11 pip
mamba activate trace-tad
```

```bash
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
```

Then install TRACE and download the model weights:

```bash
python -m pip install trace-tad
trace prepare --weights all
```

Training, evaluation, and prediction require a CUDA-capable PyTorch
environment. The annotation app can still be used for labeling workflows without
running model jobs locally.

## Usage

Start with the UI for the normal workflow:

```bash
trace app
```

The UI can annotate videos, train models, run inference, and generate matching
CLI commands for reproducible runs or batch jobs.

For command-line workflows, use the generated commands or write them directly:

```bash

# Check whether PyPI has a newer TRACE release
trace update

# Train from selected video/annotation pairs
trace train --model large --work-dir /my/dataset --pairs video01.mp4=video01_final.csv video02.mp4=video02.csv

# Evaluate the training artifact, or evaluate on held-out video/annotation pairs
trace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS
trace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --work-dir /my/testset --pairs video03.mp4=video03.csv

# Predict on new videos and write annotation drafts
trace predict --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --input /path/to/video.mp4 --annotated-video --threshold 0.25

# Run a configured end-to-end workflow
trace pipeline configs/small.py --export csv
```

`--pairs` is explicit: each item is `VIDEO_PATH=CSV_PATH`. A source video can
have multiple annotation CSVs beside it, such as `video01_draft.csv`,
`video01_final.csv`, or a model-generated `video01_predictions.csv`; each
training or evaluation pair chooses the annotation file to use for that video.
Relative paths are resolved against `--work-dir`, so
`video01.mp4=video01_final.csv` means both files are inside the work directory.
Training creates a self-contained `model_YYYYMMDD_HHMMSS/` folder under
`--work-dir`.

TRACE annotation CSVs are time-based:

```csv
labelId,timestamp,endTimestamp
grooming,12.430,18.970
rearing,42.100,45.650
```

Prediction creates a `predict_YYYYMMDD_HHMMSS/` folder beside the selected input
video or input video directory, unless `--output` is provided. The CSV output
uses the annotation format, so a reviewer can treat model predictions as draft
annotation files. Use `--annotated-video` to render MP4 overlays with the same
confidence threshold as the JSON/CSV outputs.

## Development

```bash
# Start backend and Vite dev servers
trace dev serve

# Build the frontend and bundle it into the Python package
trace dev build-frontend
```

The committed public website in [site/](site/) is intentionally small and
publishable; deeper engineering notes stay under [docs/](docs/).

## License

Apache 2.0. See [LICENSE](LICENSE).
