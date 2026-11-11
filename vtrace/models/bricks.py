import copy
import math
import torch.nn.functional as F
import torch.nn as nn
from .builder import MODELS


@MODELS.register_module()
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,  # default to none to remind, act_cfg=dict(type="relu"),
    ):
        super().__init__()
        # norm config
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.with_norm = norm_cfg is not None

        # conv config
        conv_cfg_base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.with_norm:
            conv_cfg_base["bias"] = False  # bias is not necessary with a normalization layer

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        if conv_cfg is not None:  # update conv_cfg_base
            conv_cfg_base.update(conv_cfg)

        # build conv layer
        self.conv = nn.Conv1d(**conv_cfg_base)

        # build norm layer
        if self.with_norm:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type

            if norm_type == "BN":
                self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        # build activation layer
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.with_act = act_cfg is not None

        if self.with_act:
            act_cfg = copy.copy(act_cfg)  # make a copy
            act_type = act_cfg["type"]
            act_cfg.pop("type")

            if act_type == "relu":
                self.act = nn.ReLU(inplace=True, **act_cfg)
            else:  # other type
                self.act = eval(act_type)(**act_cfg)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Conv1d):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = self.conv(x)

        if mask is not None:  # masking before the norm
            if mask.shape[-1] != x.shape[-1]:
                mask = (
                    F.interpolate(mask.unsqueeze(1).to(x.dtype), size=x.size(-1), mode="nearest")
                    .squeeze(1)
                    .to(mask.dtype)
                )
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]

        if self.with_norm:
            if self.norm_type == "LN":
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        if self.with_act:
            x = self.act(x)

        if mask is not None:  # masking the output
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]
            return x, mask
        else:
            return x


# ── merged from misc.py ──
import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4, transpose=False):
        super().__init__()
        self.scale = nn.Parameter(init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True)
        self.drop_prob = drop_prob
        self.transpose = transpose  # if False, the input is B,C,T, otherwise, the input is B,T,C

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
            x = drop_path(self.scale * x, self.drop_prob, self.training)
            return x.transpose(1, 2)
        else:
            return drop_path(self.scale * x, self.drop_prob, self.training)

    def __repr__(self):
        return f"{self.__class__.__name__}(drop_prob={self.drop_prob})"


# ── merged from sgp.py ──
import torch
import torch.nn.functional as F
import torch.nn as nn



class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=3,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        k=1.5,  # k
        group=1,  # group for cnn
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # hidden dim for mlp
        path_pdrop=0.0,  # drop path rate
        act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
        downsample_type="max",
        init_conv_vars=1,  # init gaussian variance for the weight
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = nn.LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == "max":
                kernel_size, stride, padding = n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == "avg":
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                    nn.Conv1d(n_embd, n_embd, 1, 1, 0),
                )
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        B, C, T = x.shape
        x = self.downsample(x)

        out_mask = F.interpolate(
            mask.unsqueeze(1).to(x.dtype),
            size=x.shape[-1],  # use actual downsampled feature size
            mode="nearest",
        ).detach()

        out = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.squeeze(1).bool()


# ── merged from gradient_ops.py ──
import torch
from torch.autograd import Function


class GradientScale(Function):
    """Scales gradients by a constant factor in the backward pass.

    Forward pass is identity. Backward pass multiplies gradient by `scale`.
    This allows partial gradient flow (e.g., scale=0.1) instead of full detach.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(x, scale):
    """Apply gradient scaling to tensor x.

    Args:
        x: Input tensor.
        scale: Factor to multiply gradients by in backward pass.
            scale=0 is equivalent to detach(), scale=1 is identity.

    Returns:
        Tensor with same values but scaled gradients.
    """
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return GradientScale.apply(x, scale)
