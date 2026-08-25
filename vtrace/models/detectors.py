import torch
from .postprocess import load_predictions, save_predictions, batched_nms, convert_to_seconds


def _maybe_round(value, decimals):
    if decimals is None:
        return value
    return round(value, int(decimals))


class BaseDetector(torch.nn.Module):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, **kwargs)

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, **kwargs):
        # step1: inference the model
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg)

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results

    @torch.no_grad()
    def _nms_and_format(self, segments, scores, labels, num_classes, meta, post_cfg, ext_cls):
        """Shared post-processing: NMS, time conversion, external classifier, result formatting.

        Args:
            segments: [N, 2] tensor of segment proposals
            scores: [N] tensor of confidence scores
            labels: [N] tensor of class indices
            num_classes: int
            meta: dict with video metadata
            post_cfg: post-processing config
            ext_cls: external classifier or class map list

        Returns:
            list of result dicts with keys: segment, label, score
        """
        # NMS (skip if sliding window — will be done globally later)
        if not post_cfg.sliding_window and post_cfg.nms is not None:
            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

        # convert segments to seconds
        segments = convert_to_seconds(segments, meta)

        # merge with external classifier
        if isinstance(ext_cls, list):
            labels = [ext_cls[int(label.item())] for label in labels]
        else:
            video_id = meta["video_name"]
            segments, labels, scores = ext_cls(video_id, segments, scores)

        # format results
        segs_list = segments.tolist()
        scores_list = scores.tolist() if isinstance(scores, torch.Tensor) else [s.item() for s in scores]
        time_decimals = getattr(post_cfg, "result_time_decimals", 2)
        score_decimals = getattr(post_cfg, "result_score_decimals", 4)
        results_per_video = [
            dict(
                segment=[_maybe_round(s, time_decimals) for s in seg],
                label=label,
                score=_maybe_round(sc, score_decimals),
            )
            for seg, label, sc in zip(segs_list, labels, scores_list)
        ]
        return results_per_video


# ── merged from single_stage.py ──
from .builder import DETECTORS, build_backbone, build_projection, build_neck


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Backbone -> projection -> neck skeleton shared by the detectors.

    The concrete head lives in the subclass. ``DenseLocalizer`` (the only subclass) is
    DFC-only and overrides forward_train / forward_test / post_processing, so this base
    just owns the shared feature stack.
    """

    def __init__(self, backbone=None, projection=None, neck=None):
        super(SingleStageDetector, self).__init__()

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None


# ── merged from dense_localizer.py ──
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import DETECTORS
from .bricks import Scale, AffineDropPath
from .necks import SpatialFusion, DetailCNN
from .necks import SpatialAttnPool
from .heads import build_frame_cls_head


def _resolve_num_classes(num_classes, aux_frame_cls, loc_head, rpn_head):
    """Number of behavior classes, from the first source that provides it.

    Priority: explicit kwarg > aux_frame_cls block > legacy loc_head/rpn_head block.
    The legacy fallbacks let a config that still nests num_classes under the old
    localization-head block keep working after that head was removed.
    """
    for src in (num_classes,
                (aux_frame_cls or {}).get("num_classes"),
                (loc_head or {}).get("num_classes"),
                (rpn_head or {}).get("num_classes")):
        if src is not None:
            return int(src)
    raise ValueError(
        "num_classes could not be resolved. Set model.num_classes (tools auto-detect "
        "it from the dataset class_map)."
    )


@DETECTORS.register_module()
@DETECTORS.register_module(name="TriDet")  # deprecated alias (-> DenseLocalizer)
class DenseLocalizer(SingleStageDetector):
    """Backbone -> pyramid projection -> neck -> {point-localization head, dense
    per-frame classifier}.

    Renamed from ``TriDet``: the architecture has diverged from that method, and the
    output that actually matters for the behavior task is the dense per-frame branch
    (``aux_frame_cls`` / DFC) -- the localization head's proposals are dropped at
    inference. The old registry name stays available for configs written before the
    rename.
    """

    def __init__(
        self,
        projection,
        num_classes=None,
        neck=None,
        backbone=None,
        aux_frame_cls=None,
        crop_stream_reduce="max",
        spatial_fusion=None,
        spatial_attn_pool=None,
        loss_balancing=None,
        loc_head=None,
        rpn_head=None,
    ):
        # DFC-only model: the per-frame classifier (aux_frame_cls) is the whole output.
        # The old point-localization head was discarded at inference and never helped
        # the per-frame task in ablation, so it is gone. ``loc_head`` / ``rpn_head`` are
        # still accepted (and ignored) so configs from outside this checkout — which
        # still carry the head block — build instead of crashing on an unexpected kwarg.
        super(DenseLocalizer, self).__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
        )
        if loc_head is not None or rpn_head is not None:
            import warnings
            warnings.warn(
                "loc_head/rpn_head is ignored: DenseLocalizer is DFC-only and no longer "
                "builds a localization head.",
                DeprecationWarning,
                stacklevel=2,
            )
        # num_classes is resolved from (in priority order) the explicit kwarg, the
        # aux_frame_cls block, or a legacy loc_head block; tools auto-detect and write
        # model.num_classes. See vtrace.config.num_classes_cfg.
        self._num_classes = _resolve_num_classes(num_classes, aux_frame_cls, loc_head, rpn_head)

        # Detector-side spatial fusion (§4.1). When enabled the backbone returns its
        # ViT feature MAP (return_feat_map=True) and SpatialFusion pools the H/W grid
        # — fusing (V)/(D)/(F)/coord branches before pooling instead of the backbone's
        # ConvSpatialPool. None => legacy pooled-backbone path (A2 unchanged).
        self.spatial_fusion = None
        self.detail_cnn = None
        self._use_coord = False
        self._spatial_fusion_cfg = spatial_fusion
        if spatial_fusion is not None and spatial_fusion.get("enabled", False):
            sf = dict(spatial_fusion)
            sf.pop("enabled", None)
            sf.pop("lr", None)                         # consumed in get_optim_groups
            detail_cfg = sf.pop("detail", None)        # (D) branch builder cfg
            sf.pop("frame_context", None)              # (F) builder cfg — Stage 3
            self._use_coord = bool(sf.pop("coord", False))  # (3b) coord-map source from bbox
            self.spatial_fusion = SpatialFusion(**sf)
            if detail_cfg is not None:
                self.detail_cnn = DetailCNN(**dict(detail_cfg))

        # FERAL-style spatial attention pooling (learnable-query attentive pool over the
        # patch grid) — alternative to spatial mean-pool, on the feature-map path.
        self.spatial_attn_pool = None
        if spatial_attn_pool is not None and spatial_attn_pool.get("enabled", False):
            _sap_fuse = bool(spatial_attn_pool.get("fuse_mean", False))
            self.spatial_attn_pool = SpatialAttnPool(
                in_channels=projection["in_channels"] // (2 if _sap_fuse else 1),
                window_size=projection.get("max_seq_len", 768),
                num_queries=int(spatial_attn_pool.get("num_queries", 4)),
                num_heads=int(spatial_attn_pool.get("num_heads", 8)),
                fuse_mean=_sap_fuse,
            )

        # Learned temporal upsample of the backbone's tubelet-strided token grid back
        # How to aggregate per-frame DFC logits across overlap-grouped crop
        # streams (see VideoOverlapGroupCrop). "max" is MIL/noisy-or style and is
        # idempotent on merged frames (where both streams hold the same crop);
        # "logsumexp" is a soft-max; "mean" averages. Only matters when inputs
        # carry a stream dim > 1 — single-stream runs are unaffected.
        self.crop_stream_reduce = str(crop_stream_reduce).lower()
        if self.crop_stream_reduce not in ("max", "mean", "logsumexp"):
            raise ValueError(f"Unsupported crop_stream_reduce: {crop_stream_reduce!r}")

        # Multi-task loss balancing. ``loss_balancing=dict(type="uncertainty", keys=[...])``
        # replaces the plain loss sum with Kendall&Gal homoscedastic-uncertainty
        # weighting: each balanced loss L_k is combined as exp(-s_k)*L_k + s_k with a
        # learnable log-variance s_k. With the localization head gone the only loss is
        # the DFC term, so this is a no-op unless deep-supervision adds per-stage terms.
        # None (default) => plain sum.
        self._loss_balancing = None
        self._uw_keys = []
        if loss_balancing is not None and loss_balancing.get("type") == "uncertainty":
            self._loss_balancing = "uncertainty"
            self._uw_keys = list(loss_balancing.get("keys", ["aux_frame_cls_loss"]))
            self.uw_log_vars = nn.Parameter(torch.zeros(len(self._uw_keys)))

        self.aux_frame_cls_cfg = aux_frame_cls
        self.aux_frame_cls_head = None
        self.aux_frame_bg_head = None
        self._afc_multiscale = False
        self.afc_level_weights = None
        # Boundary head (BSN/BMN-style per-frame start/end probabilities) that FEEDS
        # the DFC dense classifier — the "two heads help each other" mechanism: the
        # detection/boundary head supplies per-frame boundary localization that the
        # dense head consumes to sharpen class transitions (investigation's confused
        # band). Defaults off.
        self.aux_boundary_head = None
        self._afc_boundary_head = False
        self._afc_boundary_feed = False
        self._afc_boundary_head_loss_weight = 1.0
        if aux_frame_cls is not None and aux_frame_cls.get("enabled", False):
            num_classes = self._num_classes
            in_ch = int(aux_frame_cls.get("in_channels", projection.out_channels))
            feat_ch = int(aux_frame_cls.get("feat_channels", in_ch))
            num_layers = int(aux_frame_cls.get("num_layers", 2))
            self._afc_use_bg = bool(aux_frame_cls.get("use_background", True))
            self._afc_multilabel = bool(aux_frame_cls.get("multilabel", False))
            self._afc_target_mode = aux_frame_cls.get("target_mode", "all")
            self._afc_label_smoothing = float(aux_frame_cls.get("label_smoothing", 0.0))
            self._afc_weight = float(aux_frame_cls.get("loss_weight", 1.0))
            # MixUp (FERAL-gap exp #2): blend two clips' inputs by lam~Beta(a,a) and mix
            # their aux frame-labels; aux loss = lam*L(orig) + (1-lam)*L(perm). 0 disables.
            self._mixup_alpha = float(aux_frame_cls.get("mixup_alpha", 0.0))
            self._afc_class_weight_mode = aux_frame_cls.get("class_weight_mode", None)
            # Exponent and clamp of the inv-freq class weighting. 0.5 (the default) is the
            # historical sqrt tempering; 1.0 is plain inverse frequency.
            #
            # The clamp is NOT incidental: at power 1.0 the default max of 10 can bind on
            # several classes at once and flatten them to the same weight, which LOWERS the
            # rarest class's weight -- the opposite of the intent. Raising
            # `class_weight_clamp_max` alongside the power is what actually shifts the rare
            # classes. Mean-normalisation is applied after
            # the clamp, so what the power really controls is the rare-vs-background ratio
            # rather than any class's absolute weight.
            self._afc_class_weight_power = float(aux_frame_cls.get("class_weight_power", 0.5))
            self._afc_class_weight_clamp = (
                float(aux_frame_cls.get("class_weight_clamp_min", 0.1)),
                float(aux_frame_cls.get("class_weight_clamp_max", 10.0)),
            )
            self._afc_positive_class_weight = aux_frame_cls.get("positive_class_weight", None)
            self._afc_negative_class_weight = aux_frame_cls.get("negative_class_weight", None)
            self._afc_focal_gamma = float(aux_frame_cls.get("focal_gamma", 0.0))
            # E1 duration-normalized weighting: short bouts get per-frame upweight.
            #   dict(beta=0.5, ref=<frames or "median">, clamp=[lo,hi]). Generic, no class identity.
            self._afc_duration_norm = aux_frame_cls.get("duration_norm", None)
            # E2 confusable-pair margin: hinge on (logit_gt - logit_confusable) for configured pairs.
            #   dict(pairs=[[a,b],...], margin=0.5, weight=0.5). Pairs discovered from val confusion (here attack/inv).
            self._afc_confusable = aux_frame_cls.get("confusable_pair", None)
            self._afc_background_gate = bool(aux_frame_cls.get("background_gate", False))
            self._afc_background_gate_loss_weight = float(aux_frame_cls.get("background_gate_loss_weight", 1.0))
            self._afc_background_gate_inference_weight = float(
                aux_frame_cls.get("background_gate_inference_weight", 1.0)
            )
            self._afc_background_gate_focal_gamma = float(aux_frame_cls.get("background_gate_focal_gamma", 0.0))
            self._afc_background_gate_positive_weight = float(
                aux_frame_cls.get("background_gate_positive_weight", 1.0)
            )
            self._afc_background_gate_negative_weight = float(
                aux_frame_cls.get("background_gate_negative_weight", 1.0)
            )
            self._afc_num_classes = num_classes
            if aux_frame_cls.get("score_fusion_enabled", False):
                raise ValueError(
                    "aux_frame_cls.score_fusion_enabled is no longer supported: it fused "
                    "DFC priors into the (removed) localization head's scores."
                )
            self._afc_inference_temperature = float(aux_frame_cls.get("inference_temperature", 1.0))
            self._afc_prior_type = aux_frame_cls.get(
                "inference_prior",
                "sigmoid" if self._afc_multilabel else "softmax",
            )
            self._afc_proposal_enabled = bool(
                aux_frame_cls.get("proposal_enabled", aux_frame_cls.get("inference_enabled", False))
            )
            self._afc_proposal_mode = aux_frame_cls.get("proposal_mode", "segments")
            self._afc_proposal_prior = aux_frame_cls.get("proposal_prior", self._afc_prior_type)
            self._afc_proposal_min_score = float(aux_frame_cls.get("proposal_min_score", 0.05))
            self._afc_proposal_min_len = int(aux_frame_cls.get("proposal_min_len", 2))
            self._afc_proposal_topk = int(aux_frame_cls.get("proposal_topk", 120))
            self._afc_proposal_smoothing = int(aux_frame_cls.get("proposal_smoothing", 5))
            self._afc_proposal_duration = float(aux_frame_cls.get("proposal_duration", 1.0))
            # Boundary-focused dense loss (FCOS-centerness analogue, target side):
            # upweight DFC frames within +/- boundary_band of a GT segment start/end,
            # forcing crisp transitions where the dense head is most confused
            # (the investigation confused-frame band). 1.0 => off (legacy).
            self._afc_boundary_weight = float(aux_frame_cls.get("boundary_weight", 1.0))
            self._afc_boundary_band = int(aux_frame_cls.get("boundary_band", 4))
            # BSN/BMN-style boundary head feeding the DFC (two-heads-help). boundary_head
            # builds the per-frame start/end head; boundary_feed concats its [start,end]
            # probs into the DFC input so the dense classifier sees where transitions are.
            self._afc_boundary_head = bool(aux_frame_cls.get("boundary_head", False))
            self._afc_boundary_feed = bool(aux_frame_cls.get("boundary_feed", self._afc_boundary_head))
            self._afc_boundary_head_loss_weight = float(aux_frame_cls.get("boundary_head_loss_weight", 1.0))
            out_ch = num_classes + 1 if (self._afc_use_bg and not self._afc_multilabel) else num_classes
            _feed_active = self._afc_boundary_head and self._afc_boundary_feed
            dfc_extra = 2 if _feed_active else 0
            # FERAL-style head regularization: dropout after each hidden conv (FERAL
            # uses fc_drop 0.5). 0.0 => off. Targets val>>test overfit on small train sets.
            drop = float(aux_frame_cls.get("dropout", 0.0))

            # Optional dilated temporal stack to widen the DFC head's receptive field
            # (default kernel-3 x num_layers => RF ~5 frames @ stride 1, too local for
            # motion/ordered behaviors). ``dilations=[1,2,4,8]`` => RF ~31 frames (~1s).
            dilations = aux_frame_cls.get("dilations", None)
            # Pluggable per-frame head (different temporal inductive biases). "conv"
            # (default) is bit-identical to the original DFC conv stack so existing
            # configs are unchanged; "tcn" = residual dilated MS-TCN; "attn" = temporal
            # Transformer. See models/dense_heads/frame_cls_heads.py.
            self._afc_head_type = str(aux_frame_cls.get("head_type", "conv"))
            self.aux_frame_cls_head = build_frame_cls_head(
                self._afc_head_type, in_ch + dfc_extra, feat_ch, out_ch,
                num_layers=num_layers, dilations=dilations, drop=drop,
                num_heads=int(aux_frame_cls.get("num_heads", 8)),
                routing=aux_frame_cls.get("routing", None),
                head_kwargs=aux_frame_cls.get("head_kwargs", None),
            )
            # `inv_freq_sqrt` used to derive its class frequencies from the CURRENT
            # batch, which is unstable for a rare class: when only a few percent of
            # training windows contain it, most steps at a small batch size see none of
            # it and read freq=0. The clamp and the mean-normalisation keep the damage
            # modest — the weight reads only somewhat high on those steps against the full
            # distribution, +18%, with the other six classes 9% low — but it is a
            # systematic bias that also jitters step to step. Accumulating the counts
            # removes both: the weights become a stable function of the class mix the
            # sampler delivers. Note "what the sampler delivers", not the dataset
            # prior: an oversampling sampler shifts this on purpose (v2 anchors leave
            # grooming at ~1.47x its dataset share), and that is the mix the loss
            # should be balancing against.
            #
            # Deliberately non-persistent: train.py treats a key missing from an
            # --init_weights checkpoint as fatal, so a persistent buffer would break
            # every existing init (init_nc7.pth and friends). Re-accumulating from
            # scratch each run is cheap and the estimate is stable long before the
            # warmup ends.
            self.register_buffer(
                "_afc_class_count",
                torch.zeros(num_classes + 1, dtype=torch.float64),
                persistent=False,
            )
            if self._afc_background_gate:
                self.aux_frame_bg_head = nn.Sequential(
                    nn.Conv1d(in_ch, feat_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(feat_ch, 1, kernel_size=3, padding=1),
                )
            if self._afc_boundary_head:
                # BSN-TEM: per-frame [start, end] boundary logits on the stride-1 feature.
                self.aux_boundary_head = nn.Sequential(
                    nn.Conv1d(in_ch, feat_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(feat_ch, 2, kernel_size=3, padding=1),
                )

            # Multi-scale DFC: fuse ALL pyramid levels via a learnable softmax-weighted
            # sum (each upsampled to stride-1) instead of reading only level-0, so the
            # DFC consumes the multi-scale temporal pyramid. Init biases level-0.
            self._afc_multiscale = bool(aux_frame_cls.get("multiscale", False))
            if self._afc_multiscale:
                n_lv = len(self.projection.strides)
                w0 = torch.zeros(n_lv)
                w0[0] = 2.0
                self.afc_level_weights = nn.Parameter(w0)

        # Pyramid strides used to come from the localization head's PointGenerator; they
        # are a property of the projection pyramid, so read them there now.
        self.max_seq_len = projection.max_seq_len
        strides = self.projection.strides
        assert len(projection.sgp_win_size) == len(strides)

        max_div_factor = 1
        for s, w in zip(strides, projection.sgp_win_size):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len <= self.max_seq_len:
            max_len = self.max_seq_len
        else:
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def _coord_maps(self, box, t_tok, grid):
        """Deterministic coord-map source (§4.1) from the per-frame crop box.

        ``box``: ``[B, T, 4]`` normalized full-frame ``[x0,y0,x1,y1]``. Returns
        ``[B, 6, t_tok, grid, grid]``: 4 broadcast channels for the crop's
        full-frame placement (cx, cy, w, h) so conv knows WHERE in the frame the
        crop sits, + a 2-channel within-crop (x, y) meshgrid for absolute grid
        position. No learnable params.
        """
        b = box.shape[0]
        bx = F.interpolate(box.permute(0, 2, 1).float(), size=t_tok, mode="linear", align_corners=False)
        x0, y0, x1, y1 = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]  # [B, t_tok]
        placement = torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0], dim=1)  # [B,4,t]
        placement = placement[..., None, None].expand(b, 4, t_tok, grid, grid)
        lin = torch.linspace(0.0, 1.0, grid, device=box.device)
        ys, xs = torch.meshgrid(lin, lin, indexing="ij")
        mesh = torch.stack([xs, ys], dim=0)[None, :, None].expand(b, 2, t_tok, grid, grid)
        return torch.cat([placement, mesh], dim=1)

    def _extract_feats(self, inputs, masks, pad, stream_box=None):
        """backbone -> (pad) -> projection -> neck for one crop stream."""
        if self.with_backbone:
            if self.spatial_fusion is not None:
                # [B, Cv, T_tok, G, G] ViT map -> fuse H/W grid -> [B, C, window_size]
                feat_map = self.backbone(inputs, return_feat_map=True)
                sources = [feat_map]
                if self.detail_cnn is not None:
                    # crop frames for this stream: inputs is [B, 1, 3, T, H, W]
                    sources.append(self.detail_cnn(inputs[:, 0]))
                if self._use_coord and stream_box is not None:
                    sources.append(self._coord_maps(stream_box, feat_map.shape[2], feat_map.shape[3]))
                x = self.spatial_fusion(sources)
            elif self.spatial_attn_pool is not None:
                # FERAL-style: attentive-pool the ViT patch grid -> [B, C, window_size]
                feat_map = self.backbone(inputs, return_feat_map=True)
                x = self.spatial_attn_pool(feat_map)
            else:
                x = self.backbone(inputs)
        else:
            x = inputs
        if pad:
            x, masks = self.pad_data(x, masks)
        if self.with_projection:
            x, masks = self.projection(x, masks)
        if self.with_neck:
            x, masks = self.neck(x, masks)
        return x, masks

    def _run_crop_streams(self, inputs, masks, pad, stream_boxes=None):
        """Run each overlap-grouped crop stream independently.

        With a backbone, raw video ``inputs`` is ``[B, S, C, T, H, W]`` where ``S``
        is the crop-stream dim from ``VideoOverlapGroupCrop`` (``S==1`` reproduces
        the legacy single-crop path exactly; the backbone consumes one stream at a
        time as ``num_segs=1``). Pre-extracted feature inputs ``[B, C, T]`` carry no
        stream dim and run as a single stream. Streams share frame masks, so the
        per-level masks after projection/neck are identical — stream 0's are kept.
        """
        if isinstance(stream_boxes, (list, tuple)):  # collate may keep per-sample tensors as a list
            stream_boxes = torch.stack([torch.as_tensor(b) for b in stream_boxes], dim=0)
        if inputs.dim() == 6:
            feats_list = []
            masks_ref = None
            for s in range(inputs.shape[1]):
                box_s = stream_boxes[:, s] if stream_boxes is not None else None
                feats_s, masks_s = self._extract_feats(inputs[:, s:s + 1].contiguous(), masks, pad, box_s)
                feats_list.append(feats_s)
                if s == 0:
                    masks_ref = masks_s
            return feats_list, masks_ref

        feats, masks_ref = self._extract_feats(inputs, masks, pad)
        return [feats], masks_ref

    def _reduce_streams(self, tensors):
        """Permutation-invariant aggregation of per-stream logits."""
        if len(tensors) == 1:
            return tensors[0]
        stacked = torch.stack(tensors, dim=0)  # [S, ...]
        if self.crop_stream_reduce == "max":
            return stacked.amax(dim=0)
        if self.crop_stream_reduce == "mean":
            return stacked.mean(dim=0)
        return torch.logsumexp(stacked, dim=0)

    def _dfc_input(self, feats):
        """Feature(s) fed to the DFC head. Legacy: level-0 only. Multiscale:
        learnable softmax-weighted sum of all pyramid levels (each upsampled to the
        stride-1 length) — zero extra conv params, only n_levels softmax weights."""
        if not self._afc_multiscale or self.afc_level_weights is None or len(feats) == 1:
            return feats[0]
        t0 = feats[0].shape[-1]
        w = torch.softmax(self.afc_level_weights, dim=0)
        fused = w[0] * feats[0]
        for l in range(1, len(feats)):
            fl = feats[l]
            if fl.shape[-1] != t0:
                fl = F.interpolate(fl, size=t0, mode="linear", align_corners=False)
            fused = fused + w[l] * fl
        return fused

    def _aux_stream_logits(self, feats_list):
        """Aggregated DFC class logits, optional bg logits, optional boundary logits.

        When a boundary head exists and boundary_feed is on, its per-frame
        sigmoid([start,end]) is concatenated into the DFC class-head input — the
        dense classifier consumes the boundary head's localization (two-heads-help).
        """
        # Optional BSN-TEM boundary head feeding [start,end] probs into the DFC input.
        bnd_per_stream = None
        if self.aux_boundary_head is not None:
            bnd_per_stream = [self.aux_boundary_head(f[0]) for f in feats_list]

        def _cls_in(f, i):
            x = self._dfc_input(f)
            if self._afc_boundary_feed and bnd_per_stream is not None:
                x = torch.cat([x, torch.sigmoid(bnd_per_stream[i])], dim=1)
            return x

        cls_logits = self._reduce_streams(
            [self.aux_frame_cls_head(_cls_in(f, i)) for i, f in enumerate(feats_list)]
        )
        bg_logits = None
        if self.aux_frame_bg_head is not None:
            bg_logits = self._reduce_streams(
                [self.aux_frame_bg_head(self._dfc_input(f)).squeeze(1) for f in feats_list]
            )
        bnd_logits = self._reduce_streams(bnd_per_stream) if bnd_per_stream is not None else None
        return cls_logits, bg_logits, bnd_logits

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        stream_boxes = kwargs.pop("stream_boxes", None)

        # --- MixUp (aux-head only): blend inputs along batch, mix aux frame-labels ---
        mixup = (
            self.training
            and getattr(self, "_mixup_alpha", 0.0) > 0.0
            and isinstance(inputs, torch.Tensor)
            and inputs.shape[0] > 1
            and gt_segments is not None
        )
        mix_lam, mix_segs_b, mix_labels_b = 1.0, None, None
        if mixup:
            import numpy as np
            mix_lam = float(np.random.beta(self._mixup_alpha, self._mixup_alpha))
            perm = torch.randperm(inputs.shape[0], device=inputs.device)
            inputs = mix_lam * inputs + (1.0 - mix_lam) * inputs[perm]
            pl = perm.tolist()
            mix_segs_b = [gt_segments[i] for i in pl] if isinstance(gt_segments, (list, tuple)) else gt_segments[perm]
            mix_labels_b = [gt_labels[i] for i in pl] if isinstance(gt_labels, (list, tuple)) else gt_labels[perm]

        # pad only when computing losses in eval mode (training keeps raw length)
        feats_list, masks = self._run_crop_streams(inputs, masks, pad=not self.training, stream_boxes=stream_boxes)

        if self.aux_frame_cls_head is not None:
            cls_logits, bg_logits, bnd_logits = self._aux_stream_logits(feats_list)
            aux_loss = self._aux_frame_cls_loss(
                cls_logits, bg_logits, masks, gt_segments, gt_labels
            )
            if mixup:
                aux_loss_b = self._aux_frame_cls_loss(
                    cls_logits, bg_logits, masks, mix_segs_b, mix_labels_b
                )
                aux_loss = mix_lam * aux_loss + (1.0 - mix_lam) * aux_loss_b
            losses["aux_frame_cls_loss"] = aux_loss
            if bnd_logits is not None:
                losses["aux_boundary_loss"] = self._aux_boundary_loss(bnd_logits, masks, gt_segments)

        # only key has loss will be record
        if self._loss_balancing == "uncertainty":
            cost = 0.0
            for i, k in enumerate(self._uw_keys):
                if k in losses:
                    s = self.uw_log_vars[i]
                    cost = cost + torch.exp(-s) * losses[k] + s
            # any loss not under uncertainty weighting is added raw
            for k, v in losses.items():
                if k != "cost" and k not in self._uw_keys:
                    cost = cost + v
            losses["cost"] = cost
        else:
            losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses


    def _aux_frame_cls_loss(self, logits, bg_logits, masks, gt_segments, gt_labels):
        """Dense frame-classification auxiliary loss on stride-1 features.

        This rasterizes temporal segments into per-frame labels in the same
        stride-1 feature coordinate system used by the localization head. For mutually
        exclusive labels, use softmax CE over {classes, other}; for
        co-occurring labels, set ``multilabel=True`` for per-class BCE. ``logits``
        (``[B, out_classes, T]``) and ``bg_logits`` (``[B, T]`` or None) are the
        per-frame head outputs already aggregated across crop streams.
        """
        mask0 = masks[0]
        if mask0.dim() == 3:
            mask0 = mask0.squeeze(1)
        mask0 = mask0.bool()

        bsz, _, tlen = logits.shape
        device = logits.device
        num_classes = self._afc_num_classes

        def iter_segments(batch_idx):
            segs = gt_segments[batch_idx]
            labels = gt_labels[batch_idx]
            if segs is None or segs.numel() == 0:
                return
            for seg, label in zip(segs, labels):
                start = max(0, int(math.floor(float(seg[0].item()))))
                end = min(tlen, int(math.ceil(float(seg[1].item()))))
                if end > start:
                    yield start, end, int(label.item())

        if self._afc_multilabel:
            targets = torch.zeros((bsz, num_classes, tlen), device=device)
            for b in range(bsz):
                for start, end, label in iter_segments(b):
                    targets[b, label, start:end] = 1.0
            valid_frames = mask0
            if self._afc_target_mode == "positive_only":
                valid_frames = valid_frames & (targets.sum(dim=1) > 0)
            elif self._afc_target_mode != "all":
                raise ValueError(f"Unsupported aux_frame_cls target_mode: {self._afc_target_mode!r}")
            valid = valid_frames.unsqueeze(1).expand(-1, num_classes, -1).to(logits.dtype)
            if valid.sum() <= 0:
                return logits.sum() * 0.0
            per_frame = F.binary_cross_entropy_with_logits(logits[:, :num_classes], targets, reduction="none")
            if self._afc_focal_gamma > 0:
                probs = torch.sigmoid(logits[:, :num_classes])
                pt = torch.where(targets > 0, probs, 1.0 - probs).clamp(min=1e-6, max=1.0)
                per_frame = per_frame * (1.0 - pt).pow(self._afc_focal_gamma)
            if self._afc_class_weight_mode in ("inv_freq_sqrt", "inv_freq_sqrt_batch"):
                valid_targets = targets * valid
                class_weight = self._afc_inv_freq_sqrt_weight(
                    valid_targets.sum(dim=(0, 2)), per_frame.dtype
                )
                per_frame = per_frame * class_weight.view(1, -1, 1)
            if self._afc_positive_class_weight is not None:
                pos_weight = torch.as_tensor(
                    self._afc_positive_class_weight,
                    dtype=per_frame.dtype,
                    device=per_frame.device,
                )
                if pos_weight.numel() != num_classes:
                    raise ValueError(
                        "aux_frame_cls positive_class_weight must have "
                        f"{num_classes} entries, got {pos_weight.numel()}"
                    )
                per_frame = per_frame * torch.where(
                    targets > 0,
                    pos_weight.view(1, -1, 1),
                    torch.ones_like(pos_weight).view(1, -1, 1),
                )
            if self._afc_negative_class_weight is not None:
                neg_weight = torch.as_tensor(
                    self._afc_negative_class_weight,
                    dtype=per_frame.dtype,
                    device=per_frame.device,
                )
                if neg_weight.numel() != num_classes:
                    raise ValueError(
                        "aux_frame_cls negative_class_weight must have "
                        f"{num_classes} entries, got {neg_weight.numel()}"
                    )
                per_frame = per_frame * torch.where(
                    targets > 0,
                    torch.ones_like(neg_weight).view(1, -1, 1),
                    neg_weight.view(1, -1, 1),
                )
            loss = (per_frame * valid).sum() / valid.sum().clamp(min=1.0)
            if bg_logits is not None:
                bg_targets = (targets.sum(dim=1) == 0).to(bg_logits.dtype)
                bg_valid = mask0.to(bg_logits.dtype)
                bg_loss = F.binary_cross_entropy_with_logits(bg_logits, bg_targets, reduction="none")
                if self._afc_background_gate_focal_gamma > 0:
                    bg_probs = torch.sigmoid(bg_logits)
                    bg_pt = torch.where(bg_targets > 0, bg_probs, 1.0 - bg_probs).clamp(min=1e-6, max=1.0)
                    bg_loss = bg_loss * (1.0 - bg_pt).pow(self._afc_background_gate_focal_gamma)
                bg_weight = torch.where(
                    bg_targets > 0,
                    torch.full_like(bg_loss, self._afc_background_gate_positive_weight),
                    torch.full_like(bg_loss, self._afc_background_gate_negative_weight),
                )
                bg_loss = (bg_loss * bg_weight * bg_valid).sum() / bg_valid.sum().clamp(min=1.0)
                loss = loss + self._afc_background_gate_loss_weight * bg_loss
        else:
            out_classes = logits.shape[1]
            background_idx = num_classes if self._afc_use_bg else 0
            targets = torch.full((bsz, tlen), background_idx, dtype=torch.long, device=device)
            positive_frames = torch.zeros((bsz, tlen), dtype=torch.bool, device=device)
            boundary_frames = torch.zeros((bsz, tlen), dtype=torch.bool, device=device)
            band = self._afc_boundary_band
            # E1: per-frame duration-normalized weight (short bouts upweighted). 1.0 = off / bg frames.
            dur_w = torch.ones((bsz, tlen), device=device, dtype=logits.dtype)
            if self._afc_duration_norm is not None:
                _dn = self._afc_duration_norm
                _beta = float(_dn.get("beta", 0.5))
                _ref = _dn.get("ref", "median")
                _lo, _hi = _dn.get("clamp", [0.2, 5.0])
                _lens = [e - s for b in range(bsz) for s, e, _ in iter_segments(b)]
                if isinstance(_ref, str):
                    _ref = float(sorted(_lens)[len(_lens) // 2]) if _lens else 1.0
                _ref = max(float(_ref), 1.0)
            for b in range(bsz):
                for start, end, label in iter_segments(b):
                    targets[b, start:end] = label
                    positive_frames[b, start:end] = True
                    if self._afc_duration_norm is not None:
                        _w = (_ref / max(end - start, 1)) ** _beta
                        dur_w[b, start:end] = float(min(max(_w, _lo), _hi))
                    if self._afc_boundary_weight != 1.0:
                        boundary_frames[b, max(0, start - band):min(tlen, start + band)] = True
                        boundary_frames[b, max(0, end - band):min(tlen, end + band)] = True

            flat_logits = logits.permute(0, 2, 1).reshape(-1, out_classes)
            flat_targets = targets.reshape(-1)
            valid = mask0.reshape(-1)
            if self._afc_target_mode == "positive_only":
                valid = valid & positive_frames.reshape(-1)
            elif self._afc_target_mode != "all":
                raise ValueError(f"Unsupported aux_frame_cls target_mode: {self._afc_target_mode!r}")
            flat_logits = flat_logits[valid]
            flat_targets = flat_targets[valid]
            if flat_targets.numel() == 0:
                return logits.sum() * 0.0

            class_weight = None
            if self._afc_class_weight_mode in ("inv_freq_sqrt", "inv_freq_sqrt_batch"):
                class_weight = self._afc_inv_freq_sqrt_weight(
                    torch.bincount(flat_targets, minlength=out_classes).to(flat_logits.dtype),
                    flat_logits.dtype,
                )

            use_fw = (self._afc_boundary_weight != 1.0) or (self._afc_duration_norm is not None)
            if use_fw:
                fw = torch.ones((bsz, tlen), device=device, dtype=flat_logits.dtype)
                if self._afc_boundary_weight != 1.0:
                    fw[boundary_frames] = self._afc_boundary_weight
                fw = (fw * dur_w).reshape(-1)[valid]  # E1: combine boundary + duration weights
                ce = F.cross_entropy(
                    flat_logits, flat_targets, weight=class_weight,
                    label_smoothing=self._afc_label_smoothing, reduction="none",
                )
                loss = (ce * fw).sum() / fw.sum().clamp(min=1.0)
            else:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    weight=class_weight,
                    label_smoothing=self._afc_label_smoothing,
                )

            # E2: confusable-class margin on GT frames (flat_logits/flat_targets are valid-filtered & aligned).
            if self._afc_confusable is not None:
                _cp = self._afc_confusable
                _margin = float(_cp.get("margin", 0.5))
                _cw = float(_cp.get("weight", 0.5))
                _mode = _cp.get("mode", "pairs")
                if _mode == "hardest_negative":
                    # GENERIC: margin between GT class and its top-scoring competing class, per
                    # foreground frame. No class-pair prior — auto-targets whatever is confused
                    # at each frame, so this subsumes a hand-written pair. Easy/separable frames have
                    # margin already satisfied → hinge=0 → self-focuses on the confused frames.
                    fg = flat_targets != background_idx if self._afc_use_bg else torch.ones_like(flat_targets, dtype=torch.bool)
                    if fg.any():
                        lg = flat_logits[fg]
                        tg = flat_targets[fg]
                        gt_logit = lg.gather(1, tg[:, None]).squeeze(1)
                        masked = lg.scatter(1, tg[:, None], float("-inf"))  # out-of-place; -inf the GT col
                        hardest = masked.max(dim=1).values
                        loss = loss + _cw * F.relu(_margin - (gt_logit - hardest)).mean()
                elif _mode == "confusion_ema":
                    # GENERIC + STABLE (方案B): maintain an EMA confusion matrix over training to
                    # find, per class, its most-confused partner from the AGGREGATE (low-variance),
                    # then apply the pair margin to that data-driven partner. Unlike hardest_negative
                    # (per-frame argmax = high variance, chases easy/wrong classes early), the partner
                    # here is stable and converges to the true confusable pair without any
                    # hand-specified prior.
                    _mom = float(_cp.get("momentum", 0.99))
                    _warm = int(_cp.get("warmup_steps", 200))
                    fg = flat_targets != background_idx if self._afc_use_bg else torch.ones_like(flat_targets, dtype=torch.bool)
                    if fg.any():
                        lg = flat_logits[fg]
                        tg = flat_targets[fg]
                        oc = lg.shape[1]
                        with torch.no_grad():
                            pred = lg.argmax(1)
                            batchC = torch.zeros(oc, oc, device=lg.device, dtype=lg.dtype)
                            batchC.index_put_((tg, pred), torch.ones_like(tg, dtype=lg.dtype), accumulate=True)
                            batchC = batchC / batchC.sum(1, keepdim=True).clamp(min=1.0)  # row-normalized
                            if getattr(self, "_afc_confC", None) is None or self._afc_confC.shape[0] != oc:
                                self._afc_confC = batchC.clone()
                                self._afc_conf_steps = 0
                            else:
                                self._afc_confC.mul_(_mom).add_(batchC, alpha=1.0 - _mom)
                            self._afc_conf_steps = int(getattr(self, "_afc_conf_steps", 0)) + 1
                            _C = self._afc_confC.clone()
                            _C.fill_diagonal_(-1.0)
                            partner = _C.argmax(1)  # [oc] data-driven confused partner per class
                        if self._afc_conf_steps >= _warm:
                            part = partner[tg]
                            gt_logit = lg.gather(1, tg[:, None]).squeeze(1)
                            par_logit = lg.gather(1, part[:, None]).squeeze(1)
                            loss = loss + _cw * F.relu(_margin - (gt_logit - par_logit)).mean()
                elif _mode == "confusion_topk":
                    # 方案B (strict): like confusion_ema but apply margin ONLY to the global top-k
                    # most-confused ordered pairs (gt->partner) from the EMA matrix — classes not in
                    # the top-k (e.g. easy mount) get NO margin at all (explicit exclusion, vs
                    # confusion_ema which gives every class a partner and relies on the hinge to gate).
                    _mom = float(_cp.get("momentum", 0.99))
                    _warm = int(_cp.get("warmup_steps", 200))
                    _topk = int(_cp.get("top_k", 2))
                    fg = flat_targets != background_idx if self._afc_use_bg else torch.ones_like(flat_targets, dtype=torch.bool)
                    if fg.any():
                        lg = flat_logits[fg]
                        tg = flat_targets[fg]
                        oc = lg.shape[1]
                        with torch.no_grad():
                            pred = lg.argmax(1)
                            batchC = torch.zeros(oc, oc, device=lg.device, dtype=lg.dtype)
                            batchC.index_put_((tg, pred), torch.ones_like(tg, dtype=lg.dtype), accumulate=True)
                            batchC = batchC / batchC.sum(1, keepdim=True).clamp(min=1.0)
                            if getattr(self, "_afc_confC", None) is None or self._afc_confC.shape[0] != oc:
                                self._afc_confC = batchC.clone(); self._afc_conf_steps = 0
                            else:
                                self._afc_confC.mul_(_mom).add_(batchC, alpha=1.0 - _mom)
                            self._afc_conf_steps = int(getattr(self, "_afc_conf_steps", 0)) + 1
                            _Cd = self._afc_confC.clone(); _Cd.fill_diagonal_(-1.0)
                            _topi = _Cd.flatten().topk(min(_topk, oc * oc)).indices
                            _rows = (_topi // oc); _cols = (_topi % oc)
                            partner_lookup = torch.full((oc,), -1, device=lg.device, dtype=torch.long)
                            for _r, _c in zip(_rows.tolist(), _cols.tolist()):
                                if partner_lookup[_r] < 0:
                                    partner_lookup[_r] = _c  # topk is descending → first (highest conf) wins
                        if self._afc_conf_steps >= _warm:
                            sel = partner_lookup[tg] >= 0
                            if sel.any():
                                g = tg[sel]; p = partner_lookup[g]
                                gt_logit = lg[sel].gather(1, g[:, None]).squeeze(1)
                                par_logit = lg[sel].gather(1, p[:, None]).squeeze(1)
                                loss = loss + _cw * F.relu(_margin - (gt_logit - par_logit)).mean()
                else:
                    _terms = []
                    for _a, _bb in _cp.get("pairs", []):
                        m = flat_targets == _a
                        if m.any():
                            _terms.append(F.relu(_margin - (flat_logits[m, _a] - flat_logits[m, _bb])).mean())
                        m = flat_targets == _bb
                        if m.any():
                            _terms.append(F.relu(_margin - (flat_logits[m, _bb] - flat_logits[m, _a])).mean())
                    if _terms:
                        loss = loss + _cw * torch.stack(_terms).mean()

        return loss * self._afc_weight

    def _afc_inv_freq_sqrt_weight(self, batch_counts, dtype):
        """inv-freq-sqrt class weights from the counts seen SO FAR, not this batch.

        ``batch_counts`` is this step's per-class frame count (index order = the
        head's output channels, so single-label mode includes background last).
        It is folded into a running total and the weights are computed from that
        total, which makes them stable for classes that only appear in a small
        fraction of windows. The total converges to the mix the SAMPLER delivers,
        not to the dataset prior — with a rare-class-oversampling sampler those
        differ, and the sampled mix is the one the loss should balance. Same
        formula as before otherwise: 1/sqrt(freq), clamped to [0.1, 10] and
        normalised to mean 1.

        Eval passes do not update the running total, so a val/test forward can
        never move the weights that training is using.

        ``class_weight_mode="inv_freq_sqrt_batch"`` selects the original
        batch-derived weights instead. That mode exists to keep runs from before
        this change reproducible: it is strictly noisier — a rare class's weight swings
        with nothing but which windows the sampler drew — so new work should use the
        running estimate.
        """
        n = batch_counts.numel()
        lo, hi = self._afc_class_weight_clamp
        pw = self._afc_class_weight_power
        if self._afc_class_weight_mode == "inv_freq_sqrt_batch":
            total = batch_counts.sum().clamp(min=1.0).to(torch.float64)
            freq = batch_counts.to(torch.float64) / total
            weight = ((1.0 / (freq + 1e-6)) ** pw).clamp(lo, hi)
            return (weight / weight.mean().clamp(min=1e-6)).to(dtype)
        counts = self._afc_class_count[:n]
        if self.training:
            with torch.no_grad():
                counts += batch_counts.detach().to(counts.dtype)
        total = counts.sum()
        if total <= 0:  # first step of a fresh run and this batch was empty too
            total = batch_counts.sum().clamp(min=1.0).to(counts.dtype)
            freq = batch_counts.to(counts.dtype) / total
        else:
            freq = counts / total
        weight = ((1.0 / (freq + 1e-6)) ** pw).clamp(lo, hi)
        weight = weight / weight.mean().clamp(min=1e-6)
        return weight.to(dtype)

    def _aux_boundary_loss(self, bnd_logits, masks, gt_segments):
        """BSN-TEM boundary loss: per-frame start/end BCE on GT segment boundaries
        (box of +/- boundary_band frames), pos-weighted since boundary frames are rare."""
        mask0 = masks[0]
        if mask0.dim() == 3:
            mask0 = mask0.squeeze(1)
        mask0 = mask0.bool()
        B, _, T = bnd_logits.shape
        device = bnd_logits.device
        targets = torch.zeros((B, 2, T), device=device, dtype=bnd_logits.dtype)
        band = self._afc_boundary_band
        for b in range(B):
            segs = gt_segments[b]
            if segs is None or segs.numel() == 0:
                continue
            for seg in segs:
                s = int(round(float(seg[0].item())))
                e = int(round(float(seg[1].item())))
                targets[b, 0, max(0, s - band):min(T, s + band + 1)] = 1.0
                targets[b, 1, max(0, e - band):min(T, e + band + 1)] = 1.0
        valid = mask0.unsqueeze(1).expand(-1, 2, -1).to(bnd_logits.dtype)
        if valid.sum() <= 0:
            return bnd_logits.sum() * 0.0
        pos = (targets * valid).sum()
        pos_weight = ((valid.sum() - pos) / pos.clamp(min=1.0)).clamp(1.0, 100.0)
        per = F.binary_cross_entropy_with_logits(
            bnd_logits, targets, reduction="none", pos_weight=pos_weight
        )
        loss = (per * valid).sum() / valid.sum().clamp(min=1.0)
        return loss * self._afc_boundary_head_loss_weight

    def _aux_frame_cls_logits(self, feats):
        x = self._dfc_input(feats)
        if self._afc_boundary_feed:
            if self.aux_boundary_head is not None:
                x = torch.cat([x, torch.sigmoid(self.aux_boundary_head(feats[0]))], dim=1)
        return self.aux_frame_cls_head(x)

    def _aux_frame_bg_logits(self, feats):
        if self.aux_frame_bg_head is None:
            return None
        return self.aux_frame_bg_head(self._dfc_input(feats)).squeeze(1)

    def _aux_frame_cls_probs(self, aux_logits, prior_type=None, bg_logits=None):
        prior_type = prior_type or self._afc_prior_type
        temperature = max(self._afc_inference_temperature, 1e-6)
        if prior_type == "sigmoid":
            probs = torch.sigmoid(aux_logits[:, : self._afc_num_classes] / temperature)
        elif prior_type == "softmax":
            probs = torch.softmax(aux_logits / temperature, dim=1)[:, : self._afc_num_classes]
        else:
            raise ValueError(f"Unsupported aux_frame_cls prior type: {prior_type!r}")
        if self._afc_background_gate and bg_logits is not None:
            keep_prob = (1.0 - torch.sigmoid(bg_logits / temperature)).clamp(min=0.0, max=1.0)
            probs = probs * keep_prob.unsqueeze(1).pow(self._afc_background_gate_inference_weight)
        return probs

    def _aux_frame_cls_segments(self, frame_scores):
        """Convert dense DFC frame scores into temporal proposals."""
        if self._afc_proposal_smoothing > 1:
            kernel = self._afc_proposal_smoothing
            pad = kernel // 2
            smoothed = F.avg_pool1d(frame_scores.t().unsqueeze(0), kernel_size=kernel, stride=1, padding=pad)
            frame_scores = smoothed.squeeze(0).t()[: frame_scores.shape[0]]

        if self._afc_proposal_mode == "dense":
            keep = frame_scores >= self._afc_proposal_min_score
            frame_idxs, labels = torch.nonzero(keep, as_tuple=True)
            if frame_idxs.numel() == 0:
                device = frame_scores.device
                return (
                    torch.empty((0, 2), device=device),
                    torch.empty((0,), device=device),
                    torch.empty((0,), dtype=torch.long, device=device),
                )

            scores = frame_scores[frame_idxs, labels]
            segments = torch.stack(
                [
                    frame_idxs.to(frame_scores.dtype),
                    frame_idxs.to(frame_scores.dtype) + self._afc_proposal_duration,
                ],
                dim=1,
            )
            if self._afc_proposal_topk > 0 and scores.numel() > self._afc_proposal_topk:
                topk = min(self._afc_proposal_topk, scores.numel())
                scores, idxs = scores.topk(topk)
                segments = segments[idxs]
                labels = labels[idxs]
            return segments, scores, labels.long()

        if self._afc_proposal_mode != "segments":
            raise ValueError(f"Unsupported aux_frame_cls proposal_mode: {self._afc_proposal_mode!r}")

        segments = []
        scores = []
        labels = []
        for cls_idx in range(frame_scores.shape[1]):
            cls_scores = frame_scores[:, cls_idx]
            keep = cls_scores >= self._afc_proposal_min_score
            if not keep.any():
                continue

            padded = F.pad(keep.to(torch.int8), (1, 1))
            changes = padded[1:] - padded[:-1]
            starts = torch.nonzero(changes == 1, as_tuple=False).flatten()
            ends = torch.nonzero(changes == -1, as_tuple=False).flatten()
            for start, end in zip(starts.tolist(), ends.tolist()):
                if end - start < self._afc_proposal_min_len:
                    continue
                score = cls_scores[start:end].max()
                segments.append([float(start), float(end)])
                scores.append(float(score))
                labels.append(cls_idx)

        if not segments:
            device = frame_scores.device
            return (
                torch.empty((0, 2), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        segments = torch.tensor(segments, dtype=frame_scores.dtype, device=frame_scores.device)
        scores = torch.tensor(scores, dtype=frame_scores.dtype, device=frame_scores.device)
        labels = torch.tensor(labels, dtype=torch.long, device=frame_scores.device)

        if self._afc_proposal_topk > 0 and scores.numel() > self._afc_proposal_topk:
            topk = min(self._afc_proposal_topk, scores.numel())
            scores, idxs = scores.topk(topk)
            segments = segments[idxs]
            labels = labels[idxs]
        return segments, scores, labels

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        # DFC-only: the per-frame classifier IS the output. (The old path also ran a
        # localization head, but with pre_nms_topk=0 it contributed zero segments, so
        # this is behaviourally identical to the surviving aux branch.)
        stream_boxes = kwargs.pop("stream_boxes", None)
        feats_list, masks = self._run_crop_streams(inputs, masks, pad=True, stream_boxes=stream_boxes)
        assert self.aux_frame_cls_head is not None, (
            "DenseLocalizer is DFC-only and needs aux_frame_cls.enabled=True."
        )
        aux_logits, aux_bg_logits, _bnd = self._aux_stream_logits(feats_list)
        aux_scores = self._aux_frame_cls_probs(aux_logits, self._afc_proposal_prior, aux_bg_logits)
        aux_scores = aux_scores.permute(0, 2, 1)  # [B, T, C]
        mask0 = masks[0]
        if mask0.dim() == 3:
            mask0 = mask0.squeeze(1)
        aux_scores = aux_scores * mask0.unsqueeze(-1).to(aux_scores.dtype)
        return (aux_scores,)

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        (aux_scores,) = predictions
        num_classes = self._num_classes

        results = {}
        for i in range(len(metas)):
            segments, scores, labels = self._aux_frame_cls_segments(aux_scores[i].detach().cpu())
            video_id = metas[i]["video_name"]
            results_per_video = self._nms_and_format(
                segments, scores, labels.long(), num_classes, metas[i], post_cfg, ext_cls
            )
            if video_id in results:
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("scale"):
                    # SpatialFusion residual scale (bare nn.Parameter) — no decay
                    no_decay.add(fpn)
                elif pn.endswith("log_vars"):
                    # uncertainty-weighting learnable log-variances — no decay
                    no_decay.add(fpn)
                elif pn.endswith("level_weights"):
                    # multi-scale DFC learnable per-level softmax weights — no decay
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # Robustness: any parameter not matched by the rules above (e.g. novel pluggable-
        # head params such as SSM state tensors or other bare nn.Parameters) defaults to
        # NO weight decay rather than crashing the optimizer. Matches the common
        # "decay only Linear/Conv weights" convention and lets new heads drop in safely.
        leftover = param_dict.keys() - union_params
        if leftover:
            no_decay |= leftover
            union_params = decay | no_decay

        # Optional dedicated LR for the detector-side spatial fusion (SpatialFusion /
        # DetailCNN). The backbone's ConvSpatialPool trained at a higher lr (5e-4)
        # than the head default (1e-4); set spatial_fusion.lr to match for a fair
        # comparison. Pulls those params into their own group.
        fusion_lr = None
        if self._spatial_fusion_cfg is not None:
            fusion_lr = self._spatial_fusion_cfg.get("lr", None)
        fusion_names = set()
        if fusion_lr is not None:
            fusion_names = {n for n in union_params
                            if n.startswith("spatial_fusion") or n.startswith("detail_cnn")}

        decay_main = sorted(decay - fusion_names)
        nodecay_main = sorted(no_decay - fusion_names)
        optim_groups = [
            {"params": [param_dict[pn] for pn in decay_main], "weight_decay": cfg["weight_decay"]},
            {"params": [param_dict[pn] for pn in nodecay_main], "weight_decay": 0.0},
        ]
        if fusion_names:
            f_decay = sorted(decay & fusion_names)
            f_nodecay = sorted(no_decay & fusion_names)
            optim_groups += [
                {"params": [param_dict[pn] for pn in f_decay], "weight_decay": cfg["weight_decay"], "lr": float(fusion_lr)},
                {"params": [param_dict[pn] for pn in f_nodecay], "weight_decay": 0.0, "lr": float(fusion_lr)},
            ]
        return optim_groups


# Deprecated module-level aliases (old names kept importable).
TriDet = DenseLocalizer
