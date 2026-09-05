"""`--model maev2b` — VideoMAE V2 ViT-B/16 backbone with trainable adapters.

The default preset. The ViT blocks stay frozen and only the per-block temporal
adapters, the projection and the head train (see `optimizer.backbone`: backbone
lr=0 plus a custom `adapter` group at 1e-4). The frozen base
(`pretrained/vitB_videomaev2_k400.pth`, 173 MB) auto-downloads through
`vtrace/weights.py` on first use.

Dataset-agnostic on purpose. `_dataset.py` supplies placeholder paths that
`vtrace train` overrides per run (`data_path=` / `annotation_path=` /
`class_map=`), and `tools/train.py` auto-detects `num_classes` from the class
map. So this file fixes the model, the input geometry and the schedule — never a
corpus.

`configs/maev2b_distilled.py` is this same network started from V-JEPA 2
distilled adapters instead of random ones; `configs/calms21_demo.py` is this
same network pinned to the released CalMS21 demo checkpoint.
"""
_base_ = [
    "_dataset.py",
    "_model.py",
]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16
crop = 224  # VideoMAE-B native input

_train_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=2, resize=(crop, crop)),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="VideoTemporalAugment", speed_range=(0.7, 1.3), p=0.8),
    dict(type="VideoDecode"),
    dict(type="VideoBatchResize", scale=(crop, crop)),
    dict(type="VideoFlip", flip_ratio=0.5),
    dict(type="VideoRotate", max_angle=180.0, p=0.8),
    dict(type="VideoTrivialAugment"),
    dict(type="VideoFormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]
_val_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=2, resize=(crop, crop)),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="VideoDecode"),
    dict(type="VideoBatchResize", scale=(crop, crop)),
    dict(type="VideoFormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]
_test_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=2, resize=(crop, crop)),
    dict(type="LoadFrames", num_clips=1, method="sliding_window"),
    dict(type="VideoDecode"),
    dict(type="VideoBatchResize", scale=(crop, crop)),
    dict(type="VideoFormatShape", input_format="NCTHW"),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="imgs", keys=["masks"]),
]

dataset = dict(
    train=dict(
        type="BehaviorTargetedSlidingDataset",
        window_size=window_size,
        window_overlap_ratio=0.0,
        rare_phases=2,
        base_jitter=0.125,
        pipeline=_train_pipe,
    ),
    val=dict(window_size=window_size, window_overlap_ratio=0.5, pipeline=_val_pipe),
    test=dict(window_size=window_size, window_overlap_ratio=0.5, pipeline=_test_pipe),
)

model = dict(
    type="DenseLocalizer",
    crop_stream_reduce="max",
    backbone=dict(
        type="VisionTransformerAdapter",
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        norm_cfg=dict(type="LN", eps=1e-6),
        return_feat_map=True,
        with_cp=True,
        total_frames=window_size * scale_factor,
        adapter_index=list(range(12)),
        custom=dict(
            pretrain="pretrained/vitB_videomaev2_k400.pth",
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
            freeze_backbone=False,  # the ViT is frozen through the optimizer
        ),
    ),
    projection=dict(in_channels=768, input_noise=0.0005),
    aux_frame_cls=dict(
        enabled=True, in_channels=512, feat_channels=512, num_layers=2,
        use_background=True, multilabel=False, target_mode="all",
        label_smoothing=0.1, class_weight_mode="inv_freq_sqrt", loss_weight=2.0,
        dropout=0.5,
        inference_enabled=True, score_fusion_enabled=False, proposal_enabled=True,
        proposal_mode="dense", proposal_prior="softmax", proposal_min_score=1e-8,
        proposal_topk=0, proposal_smoothing=9, multiscale=True, mixup_alpha=0.8,
        head_type="routed", routing=dict(default="dyfadet", conv=[0]),
    ),
)

solver = dict(
    train=dict(batch_size=4, num_workers=6, persistent_workers=False, prefetch_factor=2),
    val=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    test=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    clip_grad_norm=1, ema=True, amp=True, amp_dtype="bfloat16",
    accumulation_steps=1, compile=False,
)
optimizer = dict(
    type="AdamW", lr=7e-5, weight_decay=0.025, paramwise=True,
    backbone=dict(lr=0, weight_decay=0,
                  custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)],
                  exclude=["backbone"]),
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
workflow = dict(logging_interval=50, checkpoint_interval=1, val_eval_interval=1,
                val_start_epoch=2, end_epoch=10)
evaluation = dict(
    type="Precision", subset="validation", tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    gt_fps=30.0, eval_fps=30.0, prediction_min_score=0.0,
    map_frame_filter="all", ap_mode="sklearn",
)

work_dir = "runs/maev2b"
