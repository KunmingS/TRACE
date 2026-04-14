import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Taken from
        https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = 0.25.
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        loss = sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma})"


@LOSSES.register_module()
class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with class-balanced re-weighting (Cui et al., CVPR 2019).

    Per-class weights: w_i = (1 - beta) / (1 - beta^{n_i}), normalized to
    average 1.0.  Falls back to standard focal loss when samples_per_class
    is not provided.

    Args:
        alpha: Positive/negative balancing factor for focal loss.
        gamma: Focusing parameter for focal loss.
        beta: Effective number of samples hyperparameter (default 0.999).
        samples_per_class: List of ints giving number of training segments
            per class.  If None, behaves like standard FocalLoss.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        beta: float = 0.999,
        samples_per_class=None,
    ):
        super(ClassBalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        if samples_per_class is not None:
            weights = self._compute_weights(samples_per_class, beta)
            self.register_buffer("class_weights", weights)
            logger = logging.getLogger("Train")
            logger.info(
                f"ClassBalancedFocalLoss: samples_per_class={samples_per_class}, "
                f"weights={weights.tolist()}"
            )
        else:
            self.class_weights = None

    @staticmethod
    def _compute_weights(samples_per_class, beta):
        samples = torch.tensor(samples_per_class, dtype=torch.float)
        effective_num = 1.0 - beta ** samples
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()  # normalize to average 1.0
        return weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        loss = sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma)

        if self.class_weights is not None:
            # class_weights shape: [num_classes] → broadcast over [N, num_classes]
            loss = loss * self.class_weights.to(loss.device)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, "
            f"beta={self.beta}, class_weights={self.class_weights})"
        )


def count_samples_per_class(ann_file, class_map, subset_name="training"):
    """Count the number of annotation segments per class in the training set.

    Args:
        ann_file: Path to the annotation JSON file.
        class_map: List of class name strings (ordered by class index).
        subset_name: Subset to count (default "training").

    Returns:
        List of ints with length == len(class_map).
    """
    with open(ann_file, "r") as f:
        database = json.load(f)["database"]

    counts = [0] * len(class_map)
    for video_info in database.values():
        if video_info.get("subset", "") not in subset_name:
            continue
        for anno in video_info.get("annotations", []):
            label = anno.get("label", "")
            if label in class_map:
                counts[class_map.index(label)] += 1

    # Ensure no zero counts (avoid division by zero)
    counts = [max(c, 1) for c in counts]
    return counts


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss
