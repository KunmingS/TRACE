from ..builder import DETECTORS
from .tridet import TriDet


@DETECTORS.register_module()
class TriDetBM(TriDet):
    """TriDet with Boundary-aware Multi-scale enhancements.

    Thin wrapper over TriDet that allows selecting the enhanced model
    via config (type="TriDetBM"). All logic differences are in the neck
    (TemporalDeformableFPN) and head (TriDetBMHead).
    """

    def __init__(self, projection, rpn_head, neck=None, backbone=None):
        super().__init__(
            projection=projection,
            rpn_head=rpn_head,
            neck=neck,
            backbone=backbone,
        )
