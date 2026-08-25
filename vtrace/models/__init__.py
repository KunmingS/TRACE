from .builder import (
    MODELS, DETECTORS, NECKS, PROJECTIONS,
    build_detector, build_backbone, build_projection, build_neck,
)
from .detectors import DenseLocalizer, TriDet, SingleStageDetector, BaseDetector
from .projections import SGPPyramidProj, TriDetProj
from .necks import FPNIdentity
from .heads import build_frame_cls_head
from .bricks import ConvModule, SGPBlock, Scale, TransformerBlock, DropPath, AffineDropPath
from .backbones import BackboneWrapper, VisionTransformerAdapter

__all__ = [
    "MODELS", "DETECTORS", "NECKS", "PROJECTIONS",
    "build_detector", "build_backbone", "build_projection", "build_neck",
    "build_frame_cls_head",
    "DenseLocalizer", "SingleStageDetector", "BaseDetector",
    "SGPPyramidProj", "FPNIdentity",
    # deprecated aliases (TriDet* -> DenseLocalizer / SGPPyramidProj)
    "TriDet", "TriDetProj",
    "ConvModule", "SGPBlock", "Scale", "TransformerBlock", "DropPath", "AffineDropPath",
    "BackboneWrapper", "VisionTransformerAdapter",
]
