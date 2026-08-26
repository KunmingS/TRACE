"""`--model vjepa2` — V-JEPA 2 ViT-L encoder, upper half fine-tuned.

Dataset-agnostic on purpose. `_dataset.py` supplies placeholder paths that the CLI
rewrites per run: prep writes `dataset.json` / `classmap.txt` — one entry per whole
video — and `vtrace train` passes them as `annotation_path=` / `class_map=` /
`data_path=` overrides. `tools/train.py` propagates those onto every split and auto-detects
`num_classes` from the class map. So this file fixes only the model, the input
geometry and the training schedule.

Weights come from HuggingFace (`facebook/vjepa2-vitl-fpc32-256-diving48`), not from
the GitHub release used by `maev2b` — `VJEPA2Model.from_pretrained` fetches them into
the HF cache on first use. Set `model.backbone.local_files_only=True` to forbid that
download once the cache is warm.

Heavier than `maev2b` in every direction: a ViT-L against ViT-B — ~300M encoder
params, 256x256 input, and
the top 12 of 24 blocks are trained. Expect roughly 3-4x the step time and VRAM.

INPUT SIZE IS NOT FREE HERE. The encoder derives its token grid from
`model.backbone.crop`, so feeding a different resolution requires changing that too:
`--input-resolution` alone rewrites the dataloader's resize and would desynchronise
the two. Override both, or leave both at 256.
"""
_base_ = [
    "_dataset.py",
    "_model.py",
]

window_size = 768  # frames per sliding window; must stay divisible by fpc
crop = 256  # V-JEPA 2 native input — keep model.backbone.crop in sync

IMAGENET_MEAN_255 = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD_255 = [0.229 * 255, 0.224 * 255, 0.225 * 255]

_train_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=4, resize=(crop, crop)),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="VideoTemporalAugment", speed_range=(0.7, 1.3), p=0.8),
    dict(type="VideoDecode"),
    dict(type="VideoBatchResize", scale=(crop, crop)),
    dict(type="VideoFlip", flip_ratio=0.5),
    dict(type="VideoTrivialAugment"),
    dict(type="VideoFormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]
_val_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=4, resize=(crop, crop)),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="VideoDecode"),
    dict(type="VideoBatchResize", scale=(crop, crop)),
    dict(type="VideoFormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]
_test_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=4, resize=(crop, crop)),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="VideoDecode"),
    dict(type="VideoBatchResize", scale=(crop, crop)),
    dict(type="VideoFormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="imgs", keys=["masks"]),
]

# Only window geometry and pipelines are overridden here; the split names, paths and
# dataset type come from `_dataset.py` and the CLI. `BehaviorTargetedSlidingDataset`
# adds anchor windows for classes that are rare *relative to this corpus's own mean*
# — on a balanced corpus it detects none and degenerates to plain sparse sampling,
# so it is a safe default. Use `dataset.train.type=PlainSlidingDataset` to opt out.
dataset = dict(
    train=dict(
        type="BehaviorTargetedSlidingDataset",
        window_size=window_size,
        window_overlap_ratio=0.0,
        rare_phases=2,
        base_jitter=0.125,
        pipeline=_train_pipe,
    ),
    val=dict(
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=_val_pipe,
    ),
    test=dict(
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=_test_pipe,
    ),
)

model = dict(
    backbone=dict(
        type="VJEPA2Backbone",
        model_id="facebook/vjepa2-vitl-fpc32-256-diving48",
        crop=crop,
        fpc=32,  # frames per encoder pass; window_size must divide by this
        tubelet=2,
        patch=16,
        total_frames=window_size,
        embed_dims=1024,
        mean=IMAGENET_MEAN_255,
        std=IMAGENET_STD_255,
        gradient_checkpointing=True,
        freeze_first_n_blocks=12,  # train the top half of the 24 blocks
        freeze_backbone=False,
        norm_eval=False,
        local_files_only=False,  # allow the first-use download from HuggingFace
    ),
    projection=dict(in_channels=1024, input_noise=0.0005),
)

solver = dict(
    train=dict(batch_size=4, num_workers=6, persistent_workers=False, prefetch_factor=2),
    val=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    test=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    clip_grad_norm=1,
    ema=True,
    amp=True,
    amp_dtype="bfloat16",
    accumulation_steps=1,
    compile=False,
)

# Unlike maev2b, where only the adapters move, the encoder itself trains here —
# at a lower LR than the head.
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.025,
    paramwise=True,
    backbone=dict(lr=4e-5, weight_decay=0.05, exclude=[]),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=2, max_epoch=10)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    # Soft-NMS merges the dense per-frame scores into behaviour bouts. The research
    # configs turn this off (iou_threshold=1.0, max_seg_num=500000) because frame-level
    # mAP is scored on the raw dense output; a reviewer opening the prediction file
    # wants bouts, not one entry per frame.
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.05,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=True,
)

# `vtrace pipeline --epochs N` rewrites scheduler.max_epoch / workflow.end_epoch;
# `vtrace train` takes them from here unless --cfg-options overrides them.
workflow = dict(
    logging_interval=50,
    checkpoint_interval=1,
    val_eval_interval=1,
    val_start_epoch=2,
    end_epoch=10,
)

work_dir = "runs/vjepa2"
