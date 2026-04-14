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
