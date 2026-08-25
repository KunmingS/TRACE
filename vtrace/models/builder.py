from ..registry import Registry
from .backbones import BackboneWrapper

MODELS = Registry("models")

# All component kinds share one registry; the aliases only mark intent at the
# registration site (@PROJECTIONS.register_module() etc.).
PROJECTIONS = MODELS
NECKS = MODELS
DETECTORS = MODELS


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)


def build_backbone(cfg):
    """Build backbone wrapper.

    Dispatches on ``type``: the default video-ViT path goes through
    ``BackboneWrapper`` (VisionTransformerAdapter etc.); ``VJEPA2Backbone`` is a
    self-contained trainable V-JEPA 2 encoder wrapper that already matches the
    BackboneWrapper forward interface (forward(frames, masks=None) -> [B,C,T]).
    """
    if dict(cfg).get("type") == "VJEPA2Backbone":
        from .backbones import VJEPA2Backbone

        kwargs = dict(cfg)
        kwargs.pop("type")
        return VJEPA2Backbone(**kwargs)
    return BackboneWrapper(cfg)


def build_projection(cfg):
    """Build projection."""
    return PROJECTIONS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)
