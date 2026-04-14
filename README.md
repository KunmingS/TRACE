# TRACE — Temporal Recognition of Animal Behaviors Captured from Video

TRACE is a temporal action detection system for animal behavior analysis in untrimmed video. 

![TRACE Annotator](docs/screenshot.png)

## Install

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ trace-tad
```

> Requires PyTorch with CUDA.

## Usage

```bash
# Annotate
trace serve

# Train
trace train --model large --dataset-path /my/dataset

# Test
trace test --model-path ./model --dataset-path /my/dataset

# Infer
trace infer --model-path ./model --input /path/to/video.mp4
```

## License

Apache 2.0. See [LICENSE](LICENSE).
