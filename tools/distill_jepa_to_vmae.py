"""Feature distillation: V-JEPA 2 (teacher) -> VideoMAE v2-B (student) adapters.

FD-style adapter distillation on mouse clips. The teacher is the frozen V-JEPA 2
ViT-L encoder ([B,1024,T]); the student is the
VideoMAE v2-B/16 K400 backbone with its ViT blocks FROZEN — only the in-backbone
temporal adapters (one per block) plus a new 768->1024 projection head are
trainable. We push the student's projected features to MATCH the teacher's
features on mouse-behaviour clips so the cheap, adapter-only student inherits V-JEPA 2's
mouse-interaction feature geometry without ever unfreezing the ViT.

Why this shape:
  * Teacher  : VJEPA2Backbone(crop=256, total_frames=768, embed=1024, frozen)
               forward([B,1,C,768,H,W]) -> [B,1024,768]  (channel-first).
  * Student  : the EXACT singlemouse VideoMAE-B VisionTransformerAdapter block
               (img 224, embed 768, depth 12, adapter on all 12 blocks,
               return_feat_map -> Reduce/Rearrange/Interpolate post-pipe) wrapped
               in BackboneWrapper -> [B,768,768]. VisionTransformerAdapter
               ._freeze_layers() (called every forward) freezes the ViT + patch
               embed and keeps ONLY adapter params requires_grad=True. A trainable
               Proj (Conv1d 768->1024 k3 + GELU + Conv1d 1024->1024) lifts the
               channel dim so student output is [B,1024,768] == teacher.
  * Per backbone we resize the incoming clip to the backbone's native spatial size
    (256 teacher / 224 student) BEFORE its forward, and temporally interpolate the
    two feature maps to a common T (both already T=768 here).
  * Loss = (1 - cos(norm_s, norm_t)).mean() + 0.5 * MSE(norm_s, norm_t), with the
    teacher detached; bf16 autocast; AdamW lr 1e-4 on (adapter + proj) only;
    grad-clip 1.0.

Data: a MOTION-segment manifest built by
tools/build_motion_manifest.py (subset 'train') — only windows where the mouse is
actively moving (pose-speed above a floor) are kept, so the teacher/student match on
real motion rather than the ~37%-idle open-field background. Frames are read directly
from the ORIGINAL videos via the virtual-clip path (source_video +
source_frame_offset) — NO labels are used. Override with --ann (e.g. MABE_ANN for the
old mabe-only corpus). Sampling is the proven sliding-window _test_pipe (window_size
768) over PlainSlidingDataset(test_mode).

This script ONLY trains the student adapters+proj; it does NOT touch train.py or
the eval engine. Run from the worktree root (it mirrors tools/train.py's
sys.path.insert so vtrace imports resolve).
"""
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vtrace.models.builder import build_backbone
from vtrace.datasets import build_dataset, build_dataloader


# --------------------------------------------------------------------------- #
# Paths / constants (mirror the configs we read).
# --------------------------------------------------------------------------- #
WINDOW_SIZE = 768
TEACHER_CROP = 256
STUDENT_CROP = 224
TEACHER_EMBED = 1024
STUDENT_EMBED = 768
_STUDENT_DEPTH = 12       # ViT-B: 12 blocks. --student large overrides to 24.
_STUDENT_HEADS = 12       # ViT-B: 12 heads.  --student large overrides to 16.

IMAGENET_MEAN_255 = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD_255 = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# ── Corpus paths. Defaults are placeholders: point them at your own copies, or
# override at launch with --ann / --class-map / --data-path / --out.
MABE_ANN = "data/mabe22_supervised/dataset.json"
MABE_CLASSMAP = "data/mabe22_supervised/classmap.txt"
# Frames are decoded from `source_video` in the json; data_path is only used to
# build a fallback <name>.mp4 path (never hit here), so point it at the source dir.
MABE_DATA_PATH = "data/mabe22/videos"

# A MOTION-segment manifest: per-window virtual clips pointing at the original videos
# via source_video + source_frame_offset, keeping only windows where the animal is
# actively moving. This is the DEFAULT distill corpus; pass --ann to override
# (e.g. MABE_ANN for the plain supervised set).
MOTION_ANN = "data/distill_motion/motion_manifest.json"

STUDENT_PRETRAIN = "pretrained/vitB_videomaev2_k400.pth"

DEFAULT_OUT = "runs/distill_jepa2vmae"

# scale_factor 1 -> chunk_num = WINDOW_SIZE // 16 (VideoMAE folds 16-frame chunks
# into the batch via the pre/post pipeline; matches singlemouse config).
_CHUNK_NUM = WINDOW_SIZE // 16


# --------------------------------------------------------------------------- #
# Teacher / student builders.
# --------------------------------------------------------------------------- #
def build_teacher():
    """Frozen V-JEPA 2 ViT-L. forward([B,1,C,T,H,W]) -> [B,1024,T]."""
    cfg = dict(
        type="VJEPA2Backbone",
        model_id="facebook/vjepa2-vitl-fpc32-256-diving48",
        crop=TEACHER_CROP, fpc=32, tubelet=2, patch=16, total_frames=WINDOW_SIZE,
        embed_dims=TEACHER_EMBED, mean=IMAGENET_MEAN_255, std=IMAGENET_STD_255,
        gradient_checkpointing=False,   # frozen + no_grad -> no checkpointing needed
        freeze_backbone=True, norm_eval=True, local_files_only=True,
    )
    teacher = build_backbone(cfg)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def build_student_backbone():
    """VideoMAE-B/16 K400 with frozen ViT + trainable adapters.

    Identical backbone block to configs/singlemouse_routed_mae_multiscale.py
    (return_feat_map ViT + Reduce/Rearrange/Interpolate post-pipe), wrapped in
    BackboneWrapper -> forward([B,1,C,T,H,W]) -> [B,768,T]. Only adapter params
    stay trainable (VisionTransformerAdapter._freeze_layers freezes the rest on
    every forward); BackboneWrapper.freeze_backbone is left False so adapter
    grads flow.
    """
    cfg = dict(
        type="VisionTransformerAdapter",
        img_size=224, patch_size=16, embed_dims=STUDENT_EMBED, depth=_STUDENT_DEPTH,
        num_heads=_STUDENT_HEADS, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.3,
        norm_cfg=dict(type="LN", eps=1e-6),
        return_feat_map=True, with_cp=True, total_frames=WINDOW_SIZE,
        adapter_index=list(range(_STUDENT_DEPTH)),
        custom=dict(
            pretrain=STUDENT_PRETRAIN,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"],
                     ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=_CHUNK_NUM),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=_CHUNK_NUM),
                dict(type="Interpolate", keys=["feats"], size=WINDOW_SIZE),
            ],
            norm_eval=False,
            freeze_backbone=False,
        ),
    )
    return build_backbone(cfg)


class DistillStudent(nn.Module):
    """VideoMAE-B backbone (frozen ViT, trainable adapters) + trainable Proj.

    forward([B,1,C,T,H,W]) -> [B,1024,T].
    """

    def __init__(self):
        super().__init__()
        self.backbone = build_student_backbone()
        # Freeze the ViT NOW so param counting / the optimizer only see adapters.
        # VisionTransformerAdapter._freeze_layers() (normally called inside its
        # forward) sets requires_grad=False on the patch embed + every non-adapter
        # submodule of every block; calling it here makes the freeze visible before
        # the first forward. It is idempotent and re-asserted on each forward.
        self.backbone.model.backbone._freeze_layers()
        # 768 -> 1024 temporal projection (operates on [B, C, T]).
        self.proj = nn.Sequential(
            nn.Conv1d(STUDENT_EMBED, TEACHER_EMBED, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(TEACHER_EMBED, TEACHER_EMBED, kernel_size=1),
        )

    def forward(self, frames, masks=None):
        feat = self.backbone(frames, masks)   # [B, 768, T]
        feat = self.proj(feat)                # [B, 1024, T]
        if masks is not None and feat.dim() == 3:
            feat = feat * masks.unsqueeze(1).detach().float()
        return feat

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
# Data.
# --------------------------------------------------------------------------- #
def _frames_only_test_pipe(crop):
    # Same sliding-window _test_pipe as the configs, but spatial size is left at
    # `crop` (resize happens here AND a no-op fast-path in the backbones). We use a
    # single shared loader at the larger crop and resize per-backbone in the train
    # step, so set crop here to the max native size (256) for fidelity.
    return [
        dict(type="PrepareVideoInfo", format="avi"),
        dict(type="VideoInit", num_threads=4, resize=(crop, crop)),
        dict(type="LoadFrames", num_clips=1, method="sliding_window"),
        dict(type="VideoDecode"),
        dict(type="VideoBatchResize", scale=(crop, crop)),
        dict(type="VideoFormatShape", input_format="NCTHW"),
        dict(type="ConvertToTensor", keys=["imgs"]),
        dict(type="Collect", inputs="imgs", keys=["masks"]),
    ]


def build_motion_loader(ann_file, class_map, data_path, batch_size, num_workers, logger):
    """Frames-only sliding-window loader over a dataset.json manifest.

    Records carry source_video + source_frame_offset (motion-segment virtual clips
    from a motion-manifest builder, or a supervised json); labels are
    ignored (filter_gt=False, test_mode=True). window_size=768 with frame=768 per
    record -> exactly one 768-frame window decoded from the original video.
    """
    cfg = dict(
        type="PlainSlidingDataset",
        ann_file=ann_file, subset_name="train", block_list=None,
        class_map=class_map, data_path=data_path,
        filter_gt=False, test_mode=True,          # frames-only, no labels needed
        feature_stride=1, sample_stride=1,
        window_size=WINDOW_SIZE, window_overlap_ratio=0.0,
        pipeline=_frames_only_test_pipe(TEACHER_CROP),
    )
    dataset = build_dataset(cfg, default_args=dict(logger=logger))
    loader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    return dataset, loader


# --------------------------------------------------------------------------- #
# Loss.
# --------------------------------------------------------------------------- #
def distill_loss(student_feat, teacher_feat):
    """Cosine + 0.5*MSE on channel-normalized features. teacher is detached."""
    t = teacher_feat.detach()
    # match T if a backbone ever returns a different temporal length.
    if student_feat.shape[-1] != t.shape[-1]:
        student_feat = F.interpolate(
            student_feat.float(), size=t.shape[-1], mode="linear", align_corners=False
        )
    s = F.normalize(student_feat.float(), dim=1)
    t = F.normalize(t.float(), dim=1)
    cos = (s * t).sum(dim=1)                      # [B, T]
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(s, t)
    return cos_loss + 0.5 * mse_loss, cos_loss.detach(), mse_loss.detach()


# --------------------------------------------------------------------------- #
# Spatial resize helper (frames -> a backbone's native crop).
# --------------------------------------------------------------------------- #
def resize_clip(frames, size):
    """frames: [B, num_segs, C, T, H, W] -> spatially bilinear-resize to size."""
    B, NS, C, T, H, W = frames.shape
    if H == size and W == size:
        return frames
    # decord delivers uint8 frames; interpolate needs float. The backbones cast +
    # normalize internally, so passing float here is consistent with their forward.
    x = frames.float().reshape(B * NS * C, T, H, W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.reshape(B, NS, C, T, size, size)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Distill V-JEPA2 -> VideoMAE-B adapters")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--ann", type=str, default=MOTION_ANN,
                    help="dataset.json manifest (default: jax+mabe motion segments; "
                         "pass MABE_ANN path for the old mabe-only corpus)")
    ap.add_argument("--classmap", type=str, default=MABE_CLASSMAP,
                    help="class_map path (labels unused in test_mode)")
    ap.add_argument("--data-path", type=str, default=MABE_DATA_PATH,
                    help="fallback data dir (never hit when records carry source_video)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--teacher-ckpt", type=str, default="",
                    help="override off-shelf diving48 teacher with a finetuned encoder "
                         "(encoder.* keys, e.g. a fine-tuned model's extracted encoder)")
    ap.add_argument("--student", type=str, default="base", choices=["base", "large"],
                    help="student capacity: base=ViT-B/768/12, large=ViT-L/1024/24 (K400)")
    ap.add_argument("--teacher-layer", type=int, default=-1,
                    help="V-JEPA2 encoder block to tap as the distill TARGET (1-indexed). "
                         "<=0 or >=depth (default -1) = full 24 blocks + final LN (original "
                         "behavior). A mid value (e.g. 16) tests the Distillation-Dynamics "
                         "fix: align the student's final feature to a MID teacher layer "
                         "instead of the hard-to-mimic distributed final-layer code. Mid "
                         "taps return the raw block output (no final layernorm).")
    return ap.parse_args()


def patch_teacher_out_layer(teacher, out_layer, logger):
    """Make teacher._run_encoder stop after `out_layer` blocks (1-indexed) and return
    the raw mid-block hidden state (NO final layernorm). out_layer <= 0 or >= depth
    leaves the original full-depth forward untouched. Instance-level method swap only
    — the shared VJEPA2Backbone class is not modified. Teacher is frozen + eval, so
    gradient checkpointing is off and this mirrors the original direct-call path."""
    import types
    depth = len(teacher.encoder.layer)
    if out_layer is None or out_layer <= 0 or out_layer >= depth:
        logger.info(f"teacher-layer={out_layer} -> full depth {depth} + final LN (unchanged)")
        return depth

    def _run_encoder_mid(self, x):
        hidden_states = self.encoder.embeddings(x)
        for i, layer_module in enumerate(self.encoder.layer):
            layer_outputs = layer_module(hidden_states, None)
            hidden_states = layer_outputs[0]
            if self.adapters is not None:
                hidden_states = self.adapters[i](hidden_states)
            if (i + 1) >= out_layer:
                break
        return hidden_states   # mid-layer tap: skip self.encoder.layernorm

    teacher._run_encoder = types.MethodType(_run_encoder_mid, teacher)
    logger.info(f"teacher-layer={out_layer}/{depth} -> MID-LAYER tap (raw block output, no final LN)")
    return out_layer


class _Logger:
    """Tee to stdout + {out}/distill.log."""

    def __init__(self, path):
        self.fh = open(path, "a", buffering=1)

    def info(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")


def save_trainable(model, path):
    """Save ONLY the trainable state (adapter + proj)."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if ("adapter" in k) or k.startswith("proj.")}
    torch.save(sd, path)
    return len(sd)


def main():
    args = parse_args()
    if args.student == "large":
        global STUDENT_EMBED, _STUDENT_DEPTH, _STUDENT_HEADS, STUDENT_PRETRAIN
        STUDENT_EMBED, _STUDENT_DEPTH, _STUDENT_HEADS = 1024, 24, 16
        STUDENT_PRETRAIN = "pretrained/vit-large-p16_videomaev2-k400.pth"
    os.makedirs(args.out, exist_ok=True)
    logger = _Logger(os.path.join(args.out, "distill.log"))
    ckpt_path = os.path.join(args.out, "distill_adapter.pth")

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    logger.info(f"torch={torch.__version__} cuda={torch.version.cuda} "
                f"device={torch.cuda.get_device_name(0)}")
    logger.info(f"args: {vars(args)}")

    # --- build models ---
    logger.info("Building teacher (V-JEPA2 ViT-L, frozen)...")
    teacher = build_teacher().to(device)
    if args.teacher_ckpt:
        _tsd = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        _msg = teacher.load_state_dict(_tsd, strict=False)
        logger.info(f"Teacher OVERRIDE <- {args.teacher_ckpt}: {len(_tsd)} tensors, "
                    f"missing={len(_msg.missing_keys)} unexpected={len(_msg.unexpected_keys)}")
    # Optional Distillation-Dynamics fix: tap a MID teacher layer instead of the final.
    patch_teacher_out_layer(teacher, args.teacher_layer, logger)
    logger.info("Building student (VideoMAE-B, frozen ViT + adapters + proj)...")
    student = DistillStudent().to(device)

    # adapter+proj are the only trainable params; everything else frozen.
    train_params = student.trainable_parameters()
    n_train = sum(p.numel() for p in train_params)
    n_total = sum(p.numel() for p in student.parameters())
    n_teacher_train = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    n_adapter = sum(p.numel() for n, p in student.named_parameters()
                    if p.requires_grad and "adapter" in n)
    n_proj = sum(p.numel() for n, p in student.named_parameters()
                 if p.requires_grad and n.startswith("proj."))
    logger.info(f"student trainable={n_train/1e6:.3f}M / total={n_total/1e6:.1f}M "
                f"(adapter={n_adapter/1e6:.3f}M proj={n_proj/1e6:.3f}M) | "
                f"teacher trainable={n_teacher_train}")

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    # --- data ---
    logger.info(f"Building motion-segment sliding-window loader (frames only) from {args.ann} ...")
    dataset, loader = build_motion_loader(
        args.ann, args.classmap, args.data_path, args.batch_size, args.num_workers, logger
    )
    logger.info(f"dataset windows={len(dataset)} batch_size={args.batch_size}")

    # --- train loop ---
    student.train()
    teacher.eval()
    step = 0
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            frames = batch["inputs"].to(device, non_blocking=True)   # [B,1,C,T,H,W]
            masks = batch["masks"].to(device, non_blocking=True)     # [B,T]

            frames_t = resize_clip(frames, TEACHER_CROP)
            frames_s = resize_clip(frames, STUDENT_CROP)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_feat = teacher(frames_t, masks)          # [B,1024,T]
                student_feat = student(frames_s, masks)              # [B,1024,T]
                loss, cos_l, mse_l = distill_loss(student_feat, teacher_feat)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
            optimizer.step()

            if step % args.log_interval == 0 or step < 5:
                rate = (step + 1) / max(1e-9, time.time() - t0)
                logger.info(
                    f"step {step:6d}/{args.max_steps} | loss {loss.item():.5f} "
                    f"(cos {cos_l.item():.5f} mse {mse_l.item():.5f}) | "
                    f"s_feat {tuple(student_feat.shape)} t_feat {tuple(teacher_feat.shape)} | "
                    f"{rate:.2f} it/s"
                )

            step += 1
            if step > 0 and step % args.save_interval == 0:
                step_path = os.path.join(args.out, f"distill_adapter_step{step}.pth")
                n = save_trainable(student, step_path)        # keep EACH interval as a distinct file
                save_trainable(student, ckpt_path)            # also update "latest" distill_adapter.pth
                logger.info(f"  saved {n} trainable tensors -> {step_path} (+latest) (step {step})")
            if step >= args.max_steps:
                done = True
                break

    n = save_trainable(student, ckpt_path)
    logger.info(f"DONE: saved {n} trainable tensors -> {ckpt_path}")


if __name__ == "__main__":
    main()
