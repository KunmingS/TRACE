_base_ = [
    "_dataset.py",
    "_model.py",
]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="VideoInit", num_threads=4, resize=(144, 144)),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="VideoTemporalAugment", speed_range=[0.8, 1.2], p=0.5),
            dict(type="VideoDecode"),
            dict(type="VideoBatchResize", scale=(144, 144)),
            dict(type="VideoFlip", flip_ratio=0.5),
            dict(type="VideoColorJitter", brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            dict(type="VideoFormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="VideoInit", num_threads=4, resize=(144, 144)),
            dict(type="LoadFrames", num_clips=1, method="random_trunc", scale_factor=scale_factor),
            dict(type="VideoDecode"),
            dict(type="VideoBatchResize", scale=(144, 144)),
            dict(type="VideoFormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="VideoInit", num_threads=4, resize=(144, 144)),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="VideoDecode"),
            dict(type="VideoBatchResize", scale=(144, 144)),
            dict(type="VideoFormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
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
            freeze_backbone=False,
        ),
    ),
    projection=dict(in_channels=1024, input_noise=0.0005),
)

solver = dict(
    train=dict(batch_size=1, num_workers=16, persistent_workers=True, prefetch_factor=4),
    val=dict(batch_size=4, num_workers=16, persistent_workers=True, prefetch_factor=4),
    test=dict(batch_size=4, num_workers=16, persistent_workers=True, prefetch_factor=4),
    clip_grad_norm=1,
    ema=True,
    amp=True,
    accumulation_steps=2,
    compile=False,
)

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
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=150)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        # min_score is a compaction threshold, not an output filter. Soft-NMS
        # drops items whose decayed score falls below it — shrinking the active
        # set and avoiding O(N²) work on items that will never reach the top
        # `max_seg_num` anyway. Long videos aggregate 100k+ proposals across
        # overlapping sliding windows; 0.05 is well below typical output cutoffs
        # (which sit around 0.25+) so outputs remain bit-identical to 0.001
        # while per-video NMS runs ~2× faster.
        min_score=0.05,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_eval_interval=5,
    val_start_epoch=5,
)

work_dir = "exps/large"
