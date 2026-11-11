# Third-party notices

V-TRACE is licensed under the Apache License 2.0 (see `LICENSE`). Some of its
source code was copied from, or adapted from, the open-source projects listed
below. All of that code has been modified for V-TRACE (renamed, merged into
fewer modules, and with the OpenMMLab dependencies removed). This file keeps
the notices their licences require.

Runtime libraries that V-TRACE imports but does not copy (PyTorch, NumPy,
OpenCV and so on) are installed separately and keep their own licences; they
are not covered here.

## Copied or adapted source code

| Project | Licence | Where it is used in V-TRACE |
|---|---|---|
| [OpenTAD](https://github.com/sming256/OpenTAD) — Shuming Liu et al. | Apache-2.0 | Training/validation loop, optimizers and layer-decay optimizer, LR schedulers, EMA, checkpoint utilities, sliding-window dataset, video transforms, 1-D NMS and post-processing, mAP evaluator, FPN necks (`vtrace/cores`, `vtrace/datasets`, `vtrace/models`, `vtrace/evaluations`) |
| [MMAction2](https://github.com/open-mmlab/mmaction2) / [MMCV](https://github.com/open-mmlab/mmcv) — OpenMMLab | Apache-2.0 | ViT-adapter backbone (`vtrace/models/backbones.py`), `ConvModule` (`vtrace/models/bricks.py`), config and registry APIs (`vtrace/config.py`, `vtrace/registry.py`), video transforms |
| [PySlowFast](https://github.com/facebookresearch/SlowFast) — Meta AI Research | Apache-2.0 | `drop_path` / `DropPath` (`vtrace/models/bricks.py`) |
| [ActionFormer](https://github.com/happyharrycn/actionformer_release) — Chenlin Zhang, Jianxin Wu, Yin Li | MIT | `Scale`, `AffineDropPath`, FPN identity neck, duplicate-annotation removal in the mAP evaluator |
| [TriDet](https://github.com/dingfengshi/TriDet) — Dingfeng Shi et al. | MIT | `SGPBlock` (`vtrace/models/bricks.py`), `SGPPyramidProj` (`vtrace/models/projections.py`), detector skeleton of `DenseLocalizer` |
| [ActivityNet](https://github.com/activitynet/ActivityNet) evaluation toolkit | MIT | `segment_iou`, interpolated precision/recall and average precision (`vtrace/evaluations/mAP.py`) |
| [LogME](https://github.com/thuml/LogME) — Kaichao You et al. | MIT | LogME transferability score (`vtrace/selection/transferability.py`) |
| RankMe reference snippet — Q. Garrido et al. ([paper](https://arxiv.org/abs/2210.02885)) | not stated | Four-line SVD-entropy computation (`vtrace/selection/rankme.py`) |

## Apache License 2.0 projects

OpenTAD, MMAction2 / MMCV and PySlowFast are licensed under the Apache License,
Version 2.0. A copy of that licence is in this repository's `LICENSE` file and at
<https://www.apache.org/licenses/LICENSE-2.0>.

- MMAction2 / MMCV: Copyright 2018-2019 Open-MMLab. All rights reserved.
- PySlowFast: Copyright (c) Facebook, Inc. and its affiliates.
- OpenTAD: Copyright (c) Shuming Liu and the OpenTAD contributors.

## MIT License projects

- ActionFormer: Copyright (c) 2021 University of Wisconsin-Madison
- TriDet: Copyright (c) 2023 Dingfeng Shi
- ActivityNet: Copyright (c) 2015 ActivityNet
- LogME: Copyright (c) 2023 youkaichao

Each of the above is distributed under the following licence:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
