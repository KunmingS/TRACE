"""CalMS21 downstream — VideoMAE-B/16 student whose adapters were FD-distilled
from V-JEPA 2 (see tools/distill_jepa_to_vmae.py).

Clones the no-val CalMS21 recipe from `calms21_HPSWEEP_v2_BASE.py` (train on all 70
train videos, eval on the 19 official-test videos; routed default=dyfadet head +
multiscale DFC; frame mAP_all_frames, sklearn AP; NMS off / pre_nms_topk=0) but
swaps the V-JEPA2 ViT-L backbone for the VideoMAE-B/16 `VisionTransformerAdapter`
— the EXACT student backbone block from configs/singlemouse_routed_mae_multiscale.py
and tools/distill_jepa_to_vmae.py (img 224, embed 768, depth 12, adapters on all 12
blocks). The ViT blocks stay frozen (VisionTransformerAdapter._freeze_layers, run
every forward); ONLY the in-backbone temporal adapters are trained downstream
(optimizer.backbone lr=0 + custom adapter group lr=1e-4, exactly as singlemouse).

────────────────────────────────────────────────────────────────────────────
INITIALIZING THE DISTILLED ADAPTERS
────────────────────────────────────────────────────────────────────────────
The distill run writes its trainable state to
  {out}/distill_adapter.pth
as a flat state_dict whose keys are exactly:
  backbone.model.backbone.blocks.<i>.adapter.*   (108 tensors, one set/block)
  proj.0.* / proj.2.*                            (4 tensors — the 768->1024 head)

In THIS downstream model the backbone lives at `backbone.model.backbone.*`, so the
adapter keys above map ONE-TO-ONE onto this model's state_dict (no remap needed).
The `proj.*` keys have NO home here (downstream uses TriDetProj on the raw 768-d
features, not the distillation projection) and are simply dropped.

train.py / eval_engine are NOT modified. Pick ONE of these to load the adapters:

(A) --cfg-options field + a tiny one-time pre-load step (recommended). This config
    carries `distill_adapter_ckpt` (default below). Because train.py builds the
    model and only loads weights via --resume, the simplest no-train.py-edit path
    is to bake the distilled adapters into a full TriDet checkpoint ONCE, then
    --resume from it:

      python - <<'PY'
      import torch, os, sys
      sys.path.insert(0, ".")                       # mirror tools/train.py
      from vtrace.config import Config
      from vtrace.models import build_detector
      cfg = Config.fromfile("configs/calms21_distill_vmaeB.py")
      cfg.model.num_classes = 3
      model = build_detector(cfg.model)
      ad = torch.load(cfg.distill_adapter_ckpt, map_location="cpu")
      ad = {k: v for k, v in ad.items() if "adapter" in k}   # drop proj.*
      missing, unexpected = model.load_state_dict(ad, strict=False)
      print("loaded adapters:", len(ad), "unexpected:", unexpected[:3])
      out = os.path.join(cfg.work_dir, "init_with_distilled_adapters.pth")
      os.makedirs(cfg.work_dir, exist_ok=True)
      torch.save({"epoch": -1, "state_dict": model.state_dict()}, out)
      print("wrote", out)
      PY

    then:
      python tools/train.py configs/calms21_distill_vmaeB.py \
        --resume <work_dir>/init_with_distilled_adapters.pth
    (train.py's --resume sets resume_epoch from epoch=-1 -> starts at epoch 0, and
     load_state_dict(strict=False) tolerates the adapter-only init.)

(B) Add a `load_from` hook to train.py (NOT done here, since we must not edit
    train.py): a 3-line block right after `model = build_detector(cfg.model)` that,
    when `cfg.get("distill_adapter_ckpt")` is set, loads the adapter-only keys with
    strict=False. Documented here for whoever later wants it in-engine.

If `distill_adapter_ckpt` is left unused, the adapters start from random init and
this config is just a plain frozen-ViT-B + trained-adapter CalMS21 baseline.
"""
_base_ = ["_model.py"]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16
crop = 224  # VideoMAE-B native input

# NO-VAL protocol (per calms21_HPSWEEP_v2_BASE): train on all 70 train videos, eval
# on the 19 official-test videos so mAP is comparable to the V-JEPA2 anchors.
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

# Path to the distilled trainable state written by tools/distill_jepa_to_vmae.py.
# Consumed by the one-time bake-in step (A) above; train.py itself ignores unknown
# top-level keys.
distill_adapter_ckpt = "runs/distill_jepa2vmae/distill_adapter.pth"

model = dict(
    type="DenseLocalizer",
    crop_stream_reduce="max",
    backbone=dict(
        type="VisionTransformerAdapter",
        img_size=224, patch_size=16, embed_dims=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, drop_path_rate=0.3, norm_cfg=dict(type="LN", eps=1e-6),
        return_feat_map=True, with_cp=True, total_frames=window_size * scale_factor,
        adapter_index=list(range(12)),
        custom=dict(
            # The distilled adapters are loaded SEPARATELY (see header); this is the
            # base K400 ViT-B the distillation also started from.
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
            freeze_backbone=False,   # ViT frozen via _freeze_layers; adapters trainable
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
        window_size=window_size, window_overlap_ratio=0.0, rare_phases=2,
        base_jitter=0.125, pipeline=_train_pipe),
    val=dict(type="PlainSlidingDataset", ann_file=ann, subset_name="validation", block_list=None,
        class_map=class_map, data_path=vid_test, filter_gt=False, feature_stride=1, sample_stride=1,
        window_size=window_size, window_overlap_ratio=0.5, pipeline=_test_pipe),
    test=dict(type="PlainSlidingDataset", ann_file=ann, subset_name="validation", block_list=None,
        class_map=class_map, data_path=vid_test, filter_gt=False, test_mode=True, feature_stride=1, sample_stride=1,
        window_size=window_size, window_overlap_ratio=0.5, pipeline=_test_pipe),
)
solver = dict(
    train=dict(batch_size=4, num_workers=6, persistent_workers=False, prefetch_factor=2),
    val=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    test=dict(batch_size=1, num_workers=4, persistent_workers=False, prefetch_factor=2),
    clip_grad_norm=1, ema=True, amp=True, amp_dtype="bfloat16", accumulation_steps=1, compile=False,
)
# Frozen ViT + trained adapters: backbone lr=0 except a custom adapter group at 1e-4
# (identical to configs/singlemouse_routed_mae_multiscale.py).
optimizer = dict(
    type="AdamW", lr=7e-5, weight_decay=0.025, paramwise=True,
    backbone=dict(lr=0, weight_decay=0, custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)], exclude=["backbone"]),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=2, max_epoch=10)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(pre_nms_topk=0, nms=dict(use_soft_nms=False, iou_threshold=1.0, min_score=0.0, max_seg_num=500000, multiclass=True),
    result_time_decimals=None, result_score_decimals=None, save_dict=True)
workflow = dict(logging_interval=50, checkpoint_interval=1, val_eval_interval=1, val_start_epoch=2, end_epoch=10)
evaluation = dict(type="Precision", subset="validation", tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=ann, gt_fps=30.0, eval_fps=30.0, prediction_min_score=0.0,
    map_frame_filter="all", ap_mode="sklearn")
work_dir = "runs/calms21_distill_vmaeB"
