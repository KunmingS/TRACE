import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from .vit_adapter import VisionTransformerAdapter

BACKBONE_MAP = {
    "VisionTransformerAdapter": VisionTransformerAdapter,
}


def _simple_compose(transforms_cfg, pipelines_registry):
    """Build a pipeline from a list of transform configs."""
    if not transforms_cfg:
        return None
    transforms = [pipelines_registry.build(t) for t in transforms_cfg]

    def _run(results):
        for t in transforms:
            results = t(results)
        return results

    return _run


class BackboneWrapper(nn.Module):
    """Wraps a video backbone (e.g. VisionTransformerAdapter).

    Replaces the mmaction.Recognizer3D wrapper. Handles:
    - Direct instantiation of the backbone (no mmengine registry)
    - Loading pretrained weights via torch.load
    - Pixel-level normalization (formerly ActionDataPreprocessor)
    - Pre/post-processing pipelines
    - Temporal checkpointing for memory efficiency
    """

    def __init__(self, cfg):
        super().__init__()

        # Support both dict and ConfigDict
        cfg = dict(cfg)
        custom_cfg = cfg.pop("custom")
        if hasattr(custom_cfg, "__getitem__"):
            custom_cfg = dict(custom_cfg)
        else:
            custom_cfg = custom_cfg

        # ----------------------------------------------------------------
        # Build the backbone (wrapped in a holder to match TRACEv2 naming)
        # In TRACEv2, self.model = Recognizer3D which has self.backbone = ViT
        # So param names are model.backbone.blocks.X.* — this makes the
        # optimizer exclude=["backbone"] filter work correctly.
        # ----------------------------------------------------------------
        backbone_type = cfg.pop("type")
        if backbone_type not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Available: {list(BACKBONE_MAP.keys())}"
            )
        self.model = nn.Module()
        self.model.backbone = BACKBONE_MAP[backbone_type](**cfg)

        # ----------------------------------------------------------------
        # Normalization (was: ActionDataPreprocessor)
        # Default: ImageNet mean/std used by VideoMAE
        # ----------------------------------------------------------------
        mean = custom_cfg.get("mean", [123.675, 116.28, 103.53])
        std = custom_cfg.get("std", [58.395, 57.12, 57.375])
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).reshape(1, 1, 3, 1, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).reshape(1, 1, 3, 1, 1, 1),
        )

        # ----------------------------------------------------------------
        # Load pretrained weights
        # ----------------------------------------------------------------
        pretrain = custom_cfg.get("pretrain", None)
        if pretrain is not None:
            self._load_pretrained(pretrain)
        else:
            print(
                "Warning: no pretrain path provided — backbone will be randomly initialised "
                "unless weights are loaded elsewhere."
            )

        # ----------------------------------------------------------------
        # Pre/post processing pipelines
        # ----------------------------------------------------------------
        pre_pipeline_cfg = custom_cfg.get("pre_processing_pipeline", None)
        post_pipeline_cfg = custom_cfg.get("post_processing_pipeline", None)

        if pre_pipeline_cfg or post_pipeline_cfg:
            # Lazy import to avoid circular imports
            from trace_tad.datasets.builder import PIPELINES
            self.pre_processing_pipeline = _simple_compose(pre_pipeline_cfg, PIPELINES)
            self.post_processing_pipeline = _simple_compose(post_pipeline_cfg, PIPELINES)
        else:
            self.pre_processing_pipeline = None
            self.post_processing_pipeline = None

        # ----------------------------------------------------------------
        # Misc settings
        # ----------------------------------------------------------------
        self.norm_eval = custom_cfg.get("norm_eval", True)
        self.freeze_backbone = custom_cfg.get("freeze_backbone", False)
        print(f"freeze_backbone: {self.freeze_backbone}, norm_eval: {self.norm_eval}")

        self.use_temporal_checkpointing = custom_cfg.get("temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            self.temporal_checkpointing_chunk_num = custom_cfg[
                "temporal_checkpointing_chunk_num"
            ]
            self.temporal_checkpointing_chunk_dim = custom_cfg[
                "temporal_checkpointing_chunk_dim"
            ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, frames, masks=None):
        """Forward pass.

        Args:
            frames: Tensor of shape [B, num_segs, C, T, H, W]
            masks:  Tensor of shape [B, T] (bool) or None
        """
        self.set_norm_layer()

        # Normalise: (pixel - mean) / std
        frames = (frames.float() - self.mean) / self.std

        # Pre-processing pipeline
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # Flatten batch × num_segs
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()

        # Go through backbone
        if self.freeze_backbone:
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self._temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    features = self.model.backbone(frames)
        else:
            if self.use_temporal_checkpointing:
                features = self._temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self.model.backbone(frames)

        # Unflatten and pool
        if isinstance(features, (tuple, list)):
            features = torch.cat(
                [self._unflatten_and_pool(f, batches, num_segs) for f in features],
                dim=1,
            )
        else:
            features = self._unflatten_and_pool(features, batches, num_segs)

        # Apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        return features.to(torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unflatten_and_pool(self, features, batches, num_segs):
        features = features.unflatten(dim=0, sizes=(batches, num_segs))
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights with flexible key matching."""
        print(f"Loading pretrained backbone from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Extract state dict from various checkpoint formats
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip common prefixes
        for prefix in ("backbone.", "model.backbone.", "module.backbone.", "module."):
            stripped = {
                k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
            }
            if len(stripped) > 0 and len(stripped) >= len(state_dict) // 2:
                state_dict = stripped
                break

        # Remap old-style pretrained keys to match model naming
        remapped = {}
        for k, v in state_dict.items():
            k = k.replace("patch_embed.proj.", "patch_embed.projection.")
            k = k.replace(".mlp.fc1.", ".mlp.layers.0.0.")
            k = k.replace(".mlp.fc2.", ".mlp.layers.1.")
            remapped[k] = v
        state_dict = remapped

        missing, unexpected = self.model.backbone.load_state_dict(state_dict, strict=False)

        # These keys are expected to be missing from pretrained weights:
        # - pos_embed: computed via sinusoidal encoding at init
        # - adapter.*: TRACE-specific modules, randomly initialized
        # - fc_norm.*: initialized at model construction
        expected_missing = {"pos_embed", "fc_norm.weight", "fc_norm.bias"}
        real_missing = [k for k in missing
                        if k not in expected_missing and "adapter" not in k]

        loaded = len(state_dict) - len(unexpected)
        print(f"  Loaded {loaded}/{len(state_dict)} pretrained keys.")
        if real_missing:
            print(f"  WARNING — unexpected missing keys ({len(real_missing)}): {real_missing[:5]}{'...' if len(real_missing)>5 else ''}")
        if unexpected:
            print(f"  WARNING — unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    def _temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Memory-efficient temporal checkpointing."""
        def _inner(f):
            return self.model.backbone(f)

        chunks = torch.chunk(frames, chunk_num, dim=chunk_dim)
        video_feat = [cp.checkpoint(_inner, chunk, use_reentrant=False) for chunk in chunks]

        if isinstance(video_feat[0], (tuple, list)):
            return [
                torch.cat([f[idx] for f in video_feat], dim=chunk_dim)
                for idx in range(len(video_feat[0]))
            ]
        return torch.cat(video_feat, dim=chunk_dim)
