import torch
from ..utils.post_processing import load_predictions, save_predictions, batched_nms, convert_to_seconds


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
        results_per_video = [
            dict(
                segment=[round(s, 2) for s in seg],
                label=label,
                score=round(sc, 4),
            )
            for seg, label, sc in zip(segs_list, labels, scores_list)
        ]
        return results_per_video
