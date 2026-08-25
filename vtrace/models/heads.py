"""Pluggable per-frame classification heads for the DFC (aux_frame_cls) branch.

All heads share the interface ``[B, C_in, T] -> [B, out_ch, T]`` (per-frame class
logits), so they drop into the DenseLocalizer's aux_frame_cls path (both single-scale and the
multiscale per-level fusion) unchanged. They differ ONLY in temporal INDUCTIVE BIAS,
so a dataset can pick the head whose bias matches its behaviors:

  - "conv": shallow local Conv1d stack (kernel 3), optional ``dilations``. RF ~5 frames
            (or ~31 with dilations=[1,2,4,8]). Locality + fixed small RF. This is the
            ORIGINAL DFC head — kept bit-identical so existing configs are unchanged.
  - "tcn" : MS-TCN single-stage style — residual dilated Conv1d blocks with dilation
            doubling (1,2,4,8,16,32) => large multi-scale RF (~127 frames @ 6 layers),
            residual connections. Locality + deep multi-scale temporal context.
  - "attn": Transformer encoder over the temporal axis (+ sinusoidal PE). Content-
            adaptive global relations — which other frames matter is decided by content,
            not by fixed offsets. ASFormer/ActionFormer-style relational bias.

Selected via the config field ``aux_frame_cls.head_type`` ("conv" default).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvFrameHead(nn.Module):
    """Local Conv1d stack (kernel 3), optionally dilated. == original DFC head."""

    def __init__(self, in_ch, feat_ch, out_ch, num_layers=2, dilations=None, drop=0.0):
        super().__init__()
        layers, c = [], in_ch
        ds = [int(x) for x in dilations] if dilations else [1] * num_layers
        for d in ds:
            layers += [nn.Conv1d(c, feat_ch, kernel_size=3, padding=d, dilation=d), nn.GELU()]
            if drop > 0:
                layers.append(nn.Dropout(drop))
            c = feat_ch
        layers.append(nn.Conv1d(c, out_ch, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DilatedTCNFrameHead(nn.Module):
    """MS-TCN single-stage: residual dilated Conv1d blocks, dilation doubling."""

    def __init__(self, in_ch, feat_ch, out_ch, num_layers=6, drop=0.0):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat_ch, feat_ch, kernel_size=3, padding=2 ** i, dilation=2 ** i),
                nn.GELU(),
                nn.Conv1d(feat_ch, feat_ch, kernel_size=1),
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
            )
            for i in range(num_layers)
        ])
        self.out = nn.Conv1d(feat_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = x + blk(x)  # residual dilated block
        return self.out(x)


class _SelfAttnBlock(nn.Module):
    """Pre-norm transformer block built from ONLY nn.Linear + nn.LayerNorm (no
    nn.MultiheadAttention): its fused ``in_proj_weight`` is a bare Parameter that
    TRACE's paramwise weight-decay grouping (whitelist nn.Linear/Conv1d/Conv2d)
    cannot classify, which crashes the optimizer. Explicit q/k/v Linears avoid that;
    attention itself uses F.scaled_dot_product_attention.
    """

    def __init__(self, dim, num_heads, drop):
        super().__init__()
        self.nh = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.drop = nn.Dropout(drop)
        self.attn_drop = float(drop)

    def _attn(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        h, hd = self.nh, D // self.nh
        q = self.q(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v(x).view(B, T, h, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.proj(o)

    def forward(self, x):
        x = x + self.drop(self._attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class AttnFrameHead(nn.Module):
    """Transformer over the temporal axis (content-adaptive relations) + sinusoidal PE."""

    def __init__(self, in_ch, feat_ch, out_ch, num_layers=3, num_heads=8, drop=0.1):
        super().__init__()
        self.feat_ch = feat_ch
        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.blocks = nn.ModuleList([_SelfAttnBlock(feat_ch, num_heads, drop) for _ in range(num_layers)])
        self.out = nn.Conv1d(feat_ch, out_ch, kernel_size=1)

    def _pos_enc(self, T, device, dtype):
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
        idx = torch.arange(0, self.feat_ch, 2, device=device, dtype=torch.float32)
        div = torch.exp(-math.log(10000.0) * idx / self.feat_ch)
        pe = torch.zeros(T, self.feat_ch, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype).unsqueeze(0)  # [1, T, feat]

    def forward(self, x):  # x: [B, C, T]
        h = self.proj(x).transpose(1, 2)               # [B, T, feat]
        h = h + self._pos_enc(h.shape[1], h.device, h.dtype)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h.transpose(1, 2))             # [B, out_ch, T]


class _WindowedAttnBlock(nn.Module):
    """ASFormer-style: self-attention restricted to a LOCAL temporal window (band mask).
    Content-adaptive but LOCAL relations (vs AttnFrameHead's global attention)."""

    def __init__(self, dim, num_heads, window, drop):
        super().__init__()
        self.nh, self.window = num_heads, int(window)
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.drop = nn.Dropout(drop)
        self.attn_drop = float(drop)

    def _attn(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        h, hd = self.nh, D // self.nh
        q = self.q(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v(x).view(B, T, h, hd).transpose(1, 2)
        idx = torch.arange(T, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.window  # [T,T] local band
        mask = torch.zeros(T, T, device=x.device, dtype=q.dtype).masked_fill(~band, float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        x = x + self.drop(self._attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ASFormerFrameHead(nn.Module):
    """Local windowed attention with per-layer growing window (ASFormer-style)."""

    def __init__(self, in_ch, feat_ch, out_ch, num_layers=3, num_heads=8, drop=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.blocks = nn.ModuleList([
            _WindowedAttnBlock(feat_ch, num_heads, window=2 ** (i + 1), drop=drop)
            for i in range(num_layers)
        ])
        self.out = nn.Conv1d(feat_ch, out_ch, kernel_size=1)

    def forward(self, x):  # [B, C, T]
        h = self.proj(x).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h.transpose(1, 2))


class _SSMBlock(nn.Module):
    """Diagonal SSM layer (S4D-lite): the impulse response of h_t = a*h_{t-1} + x_t is a
    per-channel geometric kernel a^k, applied as a parallel depthwise CAUSAL conv; + gating
    + FFN. Linear-time long-range recurrence bias (no mamba_ssm CUDA dependency). a=σ(θ)."""

    def __init__(self, dim, drop, kmax=256):
        super().__init__()
        self.dim, self.kmax = dim, int(kmax)
        self.norm1 = nn.LayerNorm(dim)
        self.theta = nn.Parameter(torch.linspace(-4.0, 0.0, dim))  # a = sigmoid(theta) in (0,1)
        self.D = nn.Parameter(torch.ones(dim))                     # skip connection
        self.inp = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.outp = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.drop = nn.Dropout(drop)

    def _ssm(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        u = self.inp(x).transpose(1, 2)                  # [B, D, T]
        a = torch.sigmoid(self.theta)                    # [D] in (0,1)
        K = min(self.kmax, T)
        kk = torch.arange(K, device=x.device, dtype=a.dtype)
        kernel = (a[:, None] ** kk[None, :]).unsqueeze(1)  # [D,1,K] geometric impulse response
        up = F.pad(u, (K - 1, 0))                          # left-pad => causal
        y = F.conv1d(up, kernel.flip(-1), groups=D)        # [B, D, T]
        y = y + self.D[None, :, None] * u
        y = y.transpose(1, 2)                              # [B, T, D]
        return self.outp(y * torch.sigmoid(self.gate(x)))  # gated output

    def forward(self, x):
        x = x + self.drop(self._ssm(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SSMFrameHead(nn.Module):
    """Stacked diagonal-SSM layers (S4D-lite). Long-range linear-recurrence bias."""

    def __init__(self, in_ch, feat_ch, out_ch, num_layers=4, drop=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.blocks = nn.ModuleList([_SSMBlock(feat_ch, drop) for _ in range(num_layers)])
        self.out = nn.Conv1d(feat_ch, out_ch, kernel_size=1)

    def forward(self, x):  # [B, C, T]
        h = self.proj(x).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h.transpose(1, 2))


class _DynamicAggBlock(nn.Module):
    """DyFADet-style content-adaptive local aggregation: a small net predicts per-position
    softmax weights over a local window of size k FROM the features, then aggregates the
    value sequence with those input-dependent weights. Dynamic kernels (between conv and
    attention). Operates in [B, D, T]."""

    def __init__(self, dim, k=7, drop=0.0):
        super().__init__()
        self.k = int(k)
        self.norm = nn.LayerNorm(dim)
        self.weight_net = nn.Conv1d(dim, self.k, kernel_size=3, padding=1)  # k weights/pos
        self.value = nn.Conv1d(dim, dim, kernel_size=1)
        self.outp = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # x: [B, D, T]
        B, D, T = x.shape
        xn = self.norm(x.transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.weight_net(xn), dim=1)   # [B, k, T] input-dependent weights
        v = self.value(xn)                               # [B, D, T]
        pad = self.k // 2
        vp = F.pad(v, (pad, pad))                        # [B, D, T+2pad]
        agg = 0
        for j in range(self.k):
            agg = agg + w[:, j:j + 1, :] * vp[:, :, j:j + T]
        return x + self.drop(self.outp(agg))


class DyFADetFrameHead(nn.Module):
    """Stacked dynamic content-adaptive aggregation blocks (DyFADet-style)."""

    def __init__(self, in_ch, feat_ch, out_ch, num_layers=4, k=7, drop=0.0):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.blocks = nn.ModuleList([_DynamicAggBlock(feat_ch, k=k, drop=drop) for _ in range(num_layers)])
        self.out = nn.Conv1d(feat_ch, out_ch, kernel_size=1)

    def forward(self, x):  # [B, C, T]
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class RoutedFrameHead(nn.Module):
    """Mixture-of-heads: route each output CLASS to a designated head_type (per-behavior
    heads). ``routing`` maps a head_type -> list of class indices; classes not listed go
    to ``routing["default"]``. Each sub-head produces full [out_ch] logits; the final
    per-class logit is taken from that class's assigned sub-head, so each sub-head
    specializes on its assigned behaviors. Tests "different behaviors -> different heads"."""

    def __init__(self, in_ch, feat_ch, out_ch, routing, num_layers=2,
                 dilations=None, drop=0.0, num_heads=8):
        super().__init__()
        self.out_ch = int(out_ch)
        default = routing.get("default", "attn")
        assign = [default] * self.out_ch
        for ht, classes in routing.items():
            if ht == "default":
                continue
            for c in classes:
                if 0 <= int(c) < self.out_ch:
                    assign[int(c)] = ht
        self.assign = assign
        used = sorted(set(assign))
        self.heads = nn.ModuleDict({
            ht: build_frame_cls_head(ht, in_ch, feat_ch, out_ch, num_layers=num_layers,
                                     dilations=dilations, drop=drop, num_heads=num_heads)
            for ht in used
        })
        self._cols = {ht: [c for c in range(self.out_ch) if assign[c] == ht] for ht in used}

    def forward(self, x):  # [B, Cin, T] -> [B, out_ch, T]
        B, T = x.shape[0], x.shape[-1]
        outs = {ht: head(x) for ht, head in self.heads.items()}
        # match sub-head dtype (bf16 under autocast) so the index-put below doesn't
        # mismatch a fp32 destination with bf16 source.
        out = x.new_zeros(B, self.out_ch, T, dtype=next(iter(outs.values())).dtype)
        for ht, y in outs.items():
            cols = self._cols[ht]
            if cols:
                out[:, cols, :] = y[:, cols, :]
        return out


class _DyFADetTrunk(nn.Module):
    """Faithful DyFADet pyramid + dynamic cross-scale fusion, returning the LEVEL-0
    (stride-1, length-T) FEATURE [B, feat_ch, T] -- i.e. the faithful
    ``TDynClsHead`` minus its final per-level classifier.

    Replicates ``TDynClsHead.__init__``'s ``head`` stack ((num_layers-1)
    DynamicScaleFusion blocks, each with a DTFAM('others') depth-module refiner)
    and its feature-path forward, then returns level-0. No classifier is built, so
    there are NO unused parameters (DDP-safe). Builds its own ``num_levels``
    temporal pyramid via strided max-pool, exactly like the faithful head.
    """

    def __init__(self, in_ch, feat_ch, num_layers=3, kernel_size=3,
                 num_adjacent_scales=2, dyn_type="c", tau=1.5,
                 num_levels=6, scale_factor=2):
        super().__init__()
        assert num_layers - 1 > 0, "DyFADet trunk needs >=2 layers"
        assert kernel_size % 2 == 1, "kernel_size (ka) must be odd"
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        gk = {"type": "GeReTanH", "tau": tau, "init_gate": 1.0,
              "budget_loss_lambda": 1.0, "dyn_type": dyn_type}

        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.act = nn.ReLU()
        self.head = nn.ModuleList()
        for _ in range(num_layers - 1):
            # after proj, every level carries feat_ch channels -> in_dim == feat_ch
            depth = DTFAM(dim=feat_ch, o_dim=feat_ch, ka=kernel_size, conv_type="others",
                          gate_activation=gk["type"], gate_activation_kargs=gk)
            self.head.append(
                DynamicScaleFusion(
                    feat_ch, feat_ch, num_convs=1, kernel_size=kernel_size, padding=1,
                    stride=kernel_size // 2, num_groups=1,
                    num_adjacent_scales=num_adjacent_scales, depth_module=depth,
                    gate_activation=gk["type"], gate_activation_kargs=gk,
                )
            )

    def _pyramid(self, x):
        feats = [x]
        masks = [torch.ones(x.shape[0], 1, x.shape[-1], dtype=torch.bool, device=x.device)]
        cur = x
        for _ in range(1, self.num_levels):
            cur = F.max_pool1d(cur, kernel_size=self.scale_factor, stride=self.scale_factor)
            feats.append(cur)
            masks.append(torch.ones(cur.shape[0], 1, cur.shape[-1], dtype=torch.bool, device=cur.device))
        return feats, masks

    def forward(self, x):  # x: [B, in_ch, T] -> [B, feat_ch, T] (level-0 feature)
        x = self.proj(x)
        feats, masks = self._pyramid(x)
        feats = list(feats)
        for i in range(len(self.head)):
            feats, masks = self.head[i](feats, masks)
            for j in range(len(feats)):
                feats[j] = self.act(feats[j])
        return feats[0]


class _SSMTrunk(nn.Module):
    """Lite S4D (``ssm``) diagonal-SSM stack returning the FEATURE [B, feat_ch, T]
    (the ``SSMFrameHead`` body minus its final classifier). Long-range linear-
    recurrence global branch -- the cheap structured-SSM cousin of Mamba that
    matched/beat the faithful selective-scan Mamba on CalMS21 in the bake-off."""

    def __init__(self, in_ch, feat_ch, num_layers=4, drop=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, feat_ch, kernel_size=1)
        self.blocks = nn.ModuleList([_SSMBlock(feat_ch, drop) for _ in range(num_layers)])

    def forward(self, x):  # [B, in_ch, T] -> [B, feat_ch, T]
        h = self.proj(x).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return h.transpose(1, 2)


def build_frame_cls_head(head_type, in_ch, feat_ch, out_ch, num_layers=2,
                         dilations=None, drop=0.0, num_heads=8, routing=None,
                         head_kwargs=None):
    """Factory: build the per-frame head selected by ``head_type``.

    Two families of heads share the ``[B, in_ch, T] -> [B, out_ch, T]`` interface.

    DEFAULT (recommended) heads — the canonical short names map to the original
    "lite" bake-off heads. The 2026-06-09 faithful bake-off found these match or
    BEAT the paper-faithful reproductions in TRACE's dense-localizer pipeline
    (e.g. CalMS21 tcn-lite 91.70 vs faithful MS-TCN 88.47; ssm-lite 92.20 vs
    faithful Mamba 91.81), while being cheaper and stabler — so lite is the default:
      - "conv"   -> ConvFrameHead   (original DFC conv stack)
      - "attn"   -> AttnFrameHead   (temporal Transformer)
      - "tcn"/"dilated_tcn"     -> DilatedTCNFrameHead (single-stage dilated TCN)
      - "asformer"/"windowed_attn" -> ASFormerFrameHead (encoder-only windowed attn)
      - "ssm"/"s4"/"s4d"        -> SSMFrameHead (S4D-lite diagonal SSM)
      - "dyfadet"/"dynamic"     -> DyFADetFrameHead (single-scale dynamic agg)
      - "routed" -> RoutedFrameHead (per-behavior mixture-of-heads; the overall best)
    """
    head_type = (head_type or "conv").lower()

    # ---- default / recommended (lite) heads under the canonical names ----
    if head_type == "conv":
        return ConvFrameHead(in_ch, feat_ch, out_ch, num_layers=num_layers,
                             dilations=dilations, drop=drop)
    if head_type in ("attn", "transformer"):
        return AttnFrameHead(in_ch, feat_ch, out_ch, num_layers=max(num_layers, 3),
                             num_heads=num_heads, drop=drop)
    if head_type in ("tcn", "dilated_tcn", "tcn_lite"):
        return DilatedTCNFrameHead(in_ch, feat_ch, out_ch, num_layers=max(num_layers, 6), drop=drop)
    if head_type in ("asformer", "windowed_attn", "asformer_lite"):
        return ASFormerFrameHead(in_ch, feat_ch, out_ch, num_layers=max(num_layers, 3),
                                 num_heads=num_heads, drop=drop)
    if head_type in ("ssm", "s4", "s4d"):
        return SSMFrameHead(in_ch, feat_ch, out_ch, num_layers=max(num_layers, 4), drop=drop)
    if head_type in ("dyfadet", "dynamic", "dyfadet_lite"):
        return DyFADetFrameHead(in_ch, feat_ch, out_ch, num_layers=max(num_layers, 4), drop=drop)
    if head_type == "routed":
        assert routing is not None, "head_type='routed' requires aux_frame_cls.routing"
        return RoutedFrameHead(in_ch, feat_ch, out_ch, routing, num_layers=num_layers,
                               dilations=dilations, drop=drop, num_heads=num_heads)

    raise ValueError(f"unknown aux_frame_cls head_type: {head_type!r} (expected "
                     "conv|tcn|attn|asformer|ssm|dyfadet|routed)")
