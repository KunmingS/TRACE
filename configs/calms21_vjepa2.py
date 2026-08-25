"""HP-sweep BASE config — V-JEPA 2 (256/32), no-fov, BehaviorTargeted + routed + mixup.

Clone of the proven `calms21_vjepa2_NOFOV_TARGETED_attackonly.py` (no-fov ~95.0 on the
282-test). NO-VAL protocol (per user): train on all 70 train videos, eval on the 19
official-test videos — so sweep mAP is directly comparable to the 95.0/95.15 anchors. All
HP-sweep axes are driven via `--cfg-options` off this base (no per-run config files):
  optimizer.backbone.lr / optimizer.lr ............ LR axis
  model.backbone.freeze_first_n_blocks ............ #unfrozen-layers axis (24 - N)
  dataset.train.rare_classes ...................... densify axis ([0]=attack, [0,1]=attack+inv)
Eval is the full-video sliding protocol (overlap 0.5, NMS off), frame mAP all-frames (sklearn AP).
"""
_base_ = ["_model.py"]

window_size = 768
crop = 256  # V-JEPA2 (2.0) native input
IMAGENET_MEAN_255 = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD_255 = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# NO-VAL protocol (per user): train on all 70 train videos, eval on the 19 official-test
# videos — identical to the proven 95.0 no-fov line, so sweep mAP is directly comparable.
# ── Fill these in for your CalMS21 copy ──────────────────────────────────────
# The official Task-1 release; `vtrace demo download` fetches the same videos from
# CaltechDATA into `~/.trace/demo` (or `data/calms21_demo` in a checkout), and
# `full_seq/dataset.json` is the official 70/19 split in TRACE's annotation format.
# Either edit these four, or override them at launch:
#   --cfg-options ann=... vid_train=... vid_test=... class_map=...
ann = "data/calms21/full_seq/dataset.json"
vid_train = "data/calms21/videos/train"
vid_test = "data/calms21/videos/test"
class_map = "data/calms21/full_seq/classmap.txt"

model = dict(
    type="TriDet",
    crop_stream_reduce="max",
    foveation=dict(enabled=False),
    backbone=dict(
        type="VJEPA2Backbone",
        model_id="facebook/vjepa2-vitl-fpc32-256-diving48",
        crop=crop, fpc=32, tubelet=2, patch=16, total_frames=window_size,
        embed_dims=1024, mean=IMAGENET_MEAN_255, std=IMAGENET_STD_255,
        gradient_checkpointing=True, freeze_first_n_blocks=12, freeze_backbone=False,
        norm_eval=False, local_files_only=True,
    ),
    projection=dict(in_channels=1024, input_noise=0.0005),
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

_train_pipe = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="VideoInit", num_threads=4, resize=(crop, crop)),
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
dataset = dict(
    train=dict(type="BehaviorTargetedSlidingDataset", ann_file=ann, subset_name="train", block_list=None,
        class_map=class_map, data_path=vid_train, filter_gt=False, feature_stride=1, sample_stride=1,
        window_size=window_size, window_overlap_ratio=0.0, rare_classes=[0], rare_overlap=0.75, pipeline=_train_pipe),
    val=dict(type="ThumosSlidingDataset", ann_file=ann, subset_name="validation", block_list=None,
        class_map=class_map, data_path=vid_test, filter_gt=False, feature_stride=1, sample_stride=1,
        window_size=window_size, window_overlap_ratio=0.5, pipeline=_test_pipe),
    test=dict(type="ThumosSlidingDataset", ann_file=ann, subset_name="validation", block_list=None,
        class_map=class_map, data_path=vid_test, filter_gt=False, test_mode=True, feature_stride=1, sample_stride=1,
        window_size=window_size, window_overlap_ratio=0.5, pipeline=_test_pipe),
)
solver = dict(
    train=dict(batch_size=4, num_workers=6, persistent_workers=False, prefetch_factor=2),
    val=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    test=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    clip_grad_norm=1, ema=True, amp=True, amp_dtype="bfloat16", accumulation_steps=1, compile=False,
)
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.025, paramwise=True,
    backbone=dict(lr=4e-5, weight_decay=0.05, exclude=[]))
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=2, max_epoch=10)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(pre_nms_topk=0, nms=dict(use_soft_nms=False, iou_threshold=1.0, min_score=0.0, max_seg_num=500000, multiclass=True),
    result_time_decimals=None, result_score_decimals=None, save_dict=True)
workflow = dict(logging_interval=50, checkpoint_interval=1, val_eval_interval=1, val_start_epoch=2, end_epoch=10)
evaluation = dict(type="Precision", subset="validation", tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=ann, gt_fps=30.0, eval_fps=30.0, prediction_min_score=0.0,
    map_frame_filter="all", ap_mode="sklearn")
work_dir = "runs/calms21_vjepa2"
