"""`--model maev2` — VideoMAE V2 ViT-L backbone with trainable adapters.

Dataset-agnostic on purpose. `_dataset.py` supplies placeholder paths that the CLI
rewrites per run: prep writes `dataset.json` / `classmap.txt` — one entry per whole
video — and `vtrace train` passes them as `annotation_path=` / `class_map=` /
`data_path=` overrides. `tools/train.py` propagates those onto every split and auto-detects
`num_classes` from the class map. So this file fixes only the model, the input
geometry and the training schedule.

The ViT is frozen; only the per-block adapters, the projection and the head train
(see `optimizer.backbone`). The backbone weights are pulled from the project's
GitHub release on first use — `pretrained/vit-large-p16_videomaev2-k400.pth` is
resolved by basename against the registry in `vtrace/weights.py`, so the path
need not exist locally.

Input is 144x144, not the ViT's native 224: a 768-frame window is the expensive
axis here, and the positional embedding is interpolated to whatever comes in.
Raise `crop` (or pass `--input-resolution`) to trade throughput for detail.
"""
_base_ = [
    "_dataset.py",
    "_model.py",
]

window_size = 768  # frames per sliding window
scale_factor = 1
chunk_num = window_size * scale_factor // 16
crop = 144  # decode/compute size; the ViT interpolates its pos-embed to match

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
        type="VisionTransformerAdapter",
        img_size=224,
        patch_size=16,
        embed_dims=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        norm_cfg=dict(type="LN", eps=1e-6),
        return_feat_map=True,
        with_cp=True,
        total_frames=window_size * scale_factor,
        adapter_index=list(range(24)),
        custom=dict(
            pretrain="pretrained/vit-large-p16_videomaev2-k400.pth",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,
            freeze_backbone=False,  # the ViT is frozen through the optimizer, not here
        ),
    ),
    projection=dict(in_channels=1024, input_noise=0.0005),
)

solver = dict(
    train=dict(batch_size=1, num_workers=6, persistent_workers=False, prefetch_factor=2),
    val=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    test=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    clip_grad_norm=1,
    ema=True,
    amp=True,
    amp_dtype="bfloat16",
    accumulation_steps=2,  # effective train batch 2
    compile=False,
)

# Frozen ViT + trained adapters: backbone lr=0 except the adapter parameter group.
optimizer = dict(
    type="AdamW",
    lr=7e-5,
    weight_decay=0.025,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
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

work_dir = "runs/maev2"
