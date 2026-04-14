import os
import json
import time
import tqdm
import torch
import numpy as np
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from .train_engine import _to_cuda, _swap_ema_weights

from trace_tad.utils import create_folder
from trace_tad.models.utils.post_processing import (
    build_classifier, batched_nms, load_predictions, save_predictions,
)
from trace_tad.evaluations import build_evaluator
from trace_tad.datasets.base import SlidingWindowDataset


class InferenceTimer:
    """Lightweight CPU+GPU timing collector for inference profiling."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self._cpu_starts = {}

    def cpu_start(self, name):
        if not self.enabled:
            return
        self._cpu_starts[name] = time.perf_counter()

    def cpu_end(self, name):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._cpu_starts[name]
        self.timings[name] += elapsed
        self.counts[name] += 1

    def gpu_start(self, name):
        if not self.enabled:
            return
        self._gpu_start = torch.cuda.Event(enable_timing=True)
        self._gpu_end = torch.cuda.Event(enable_timing=True)
        self._gpu_start.record()
        self._current_gpu_name = name

    def gpu_end(self, name):
        if not self.enabled:
            return
        self._gpu_end.record()
        torch.cuda.synchronize()
        elapsed_ms = self._gpu_start.elapsed_time(self._gpu_end)
        self.timings[name] += elapsed_ms / 1000.0
        self.counts[name] += 1

    def log_summary(self, logger, num_batches):
        if not self.enabled:
            return
        total = sum(self.timings.values())
        if total == 0:
            return

        logger.info("=" * 80)
        logger.info("Inference Profiling Summary")
        logger.info("-" * 80)
        logger.info(f"{'Phase':<30} | {'Total (s)':>10} | {'Avg/batch (ms)':>15} | {'% of total':>10}")
        logger.info("-" * 80)

        phase_order = [
            "DataLoader",
            "GPU Transfer",
            "Forward Test (GPU)",
            "GPU→CPU Sync",
            "Post-processing (CPU)",
            "Sliding Window NMS",
            "Evaluation",
        ]
        for phase in phase_order:
            if phase in self.timings:
                t = self.timings[phase]
                avg_ms = (t / max(self.counts[phase], 1)) * 1000
                pct = (t / total) * 100
                logger.info(f"{phase:<30} | {t:>10.3f} | {avg_ms:>15.2f} | {pct:>9.1f}%")

        for phase, t in sorted(self.timings.items()):
            if phase not in phase_order:
                avg_ms = (t / max(self.counts[phase], 1)) * 1000
                pct = (t / total) * 100
                logger.info(f"{phase:<30} | {t:>10.3f} | {avg_ms:>15.2f} | {pct:>9.1f}%")

        logger.info("-" * 80)
        logger.info(f"{'TOTAL':<30} | {total:>10.3f} | {(total / max(num_batches, 1)) * 1000:>15.2f} | {'100.0%':>10}")
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MB")
        logger.info("=" * 80)


def _jsonify_metrics(value):
    """Convert NumPy-heavy evaluation outputs into plain JSON-safe types."""
    if isinstance(value, dict):
        return {key: _jsonify_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_metrics(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _predictions_to_cpu(predictions):
    """Recursively move prediction tensors to CPU."""
    if isinstance(predictions, torch.Tensor):
        return predictions.detach().cpu()
    elif isinstance(predictions, (tuple, list)):
        return type(predictions)(_predictions_to_cpu(p) for p in predictions)
    return predictions


def _nms_one_video(video_id, detections, post_cfg):
    """Run NMS for a single video's sliding window detections. Returns (video_id, nms_results)."""
    segments = torch.tensor([d["segment"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)

    class_to_idx = {}
    label_indices = []
    for d in detections:
        lbl = d["label"]
        if lbl not in class_to_idx:
            class_to_idx[lbl] = len(class_to_idx)
        label_indices.append(class_to_idx[lbl])
    labels = torch.tensor(label_indices, dtype=torch.float32)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

    segs_list = segments.tolist()
    scores_list = scores.tolist()
    labels_list = labels.tolist()
    results = [
        dict(
            segment=[round(s, 2) for s in seg],
            label=idx_to_class[int(lbl)],
            score=round(sc, 4),
        )
        for seg, lbl, sc in zip(segs_list, labels_list, scores_list)
    ]
    return video_id, results


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    model_ema=None,
    use_amp=False,
    not_eval=False,
    profile=False,
):
    """Pipelined inference: overlaps CPU post-processing and sliding window NMS with GPU forward."""

    timer = InferenceTimer(enabled=profile)

    # swap model weights with EMA weights for evaluation
    if model_ema is not None:
        _swap_ema_weights(model, model_ema)

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    is_sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)
    cfg.post_processing.sliding_window = is_sliding_window

    model.eval()
    # raw_result_dict: accumulates per-window detections (before cross-window NMS)
    # final_result_dict: stores NMS'd results (after cross-window NMS)
    raw_result_dict = {}
    final_result_dict = {}
    result_lock = threading.Lock()
    num_batches = len(test_loader)

    # Thread pool for overlapping CPU work with GPU forward
    # max_workers=2: one for post-processing, one for sliding window NMS
    executor = ThreadPoolExecutor(max_workers=2)
    pending_postproc = None
    nms_futures = []

    # Track which video_ids have been seen so we can detect completion
    # Since shuffle=False, windows for the same video arrive consecutively
    completed_videos = set()

    def _postproc_and_aggregate(preds_cpu, metas):
        """Run post-processing on CPU and aggregate results (called from thread)."""
        results = model.post_processing(preds_cpu, metas, cfg.post_processing, ext_cls=external_cls)
        with result_lock:
            for k, v in results.items():
                if k in raw_result_dict:
                    raw_result_dict[k].extend(v)
                else:
                    raw_result_dict[k] = v

    def _submit_nms_for_completed_videos(current_video_ids):
        """Check if any videos in raw_result_dict are no longer in the current batch,
        meaning all their windows have been processed. Submit their NMS."""
        if not is_sliding_window or cfg.post_processing.nms is None:
            return
        with result_lock:
            # Videos in raw_result_dict but NOT in current batch and NOT already completed
            ready_videos = set(raw_result_dict.keys()) - current_video_ids - completed_videos
            for vid in ready_videos:
                detections = raw_result_dict.pop(vid)
                completed_videos.add(vid)
                future = executor.submit(_nms_one_video, vid, detections, cfg.post_processing)
                nms_futures.append(future)

    data_iter = iter(test_loader)
    prev_video_ids = set()

    for batch_idx in tqdm.tqdm(range(num_batches)):
        # --- Phase 1: Load data ---
        timer.cpu_start("DataLoader")
        data_dict = next(data_iter)
        timer.cpu_end("DataLoader")

        timer.cpu_start("GPU Transfer")
        data_dict = _to_cuda(data_dict)
        timer.cpu_end("GPU Transfer")

        # --- Phase 2: GPU forward pass ---
        timer.gpu_start("Forward Test (GPU)")
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            with torch.inference_mode():
                if cfg.inference.load_from_raw_predictions:
                    predictions = load_predictions(data_dict["metas"], cfg.inference)
                else:
                    predictions = model.forward_test(
                        data_dict["inputs"], data_dict["masks"],
                        data_dict.get("metas"), cfg.inference,
                    )
                    if cfg.inference.save_raw_prediction:
                        save_predictions(predictions, data_dict["metas"], cfg.inference.folder)
        timer.gpu_end("Forward Test (GPU)")

        # --- Phase 3: Move predictions to CPU ---
        timer.cpu_start("GPU→CPU Sync")
        predictions_cpu = _predictions_to_cpu(predictions)
        metas = data_dict["metas"]
        timer.cpu_end("GPU→CPU Sync")

        # --- Phase 4: Wait for previous post-processing, then submit current ---
        timer.cpu_start("Post-processing (CPU)")
        if pending_postproc is not None:
            pending_postproc.result()
        timer.cpu_end("Post-processing (CPU)")

        # Determine which videos are in this batch
        current_video_ids = set(m["video_name"] for m in metas)

        # Videos from previous batch that are NOT in current batch are complete
        # → submit their sliding window NMS (runs in thread while GPU processes next batch)
        timer.cpu_start("Sliding Window NMS")
        _submit_nms_for_completed_videos(current_video_ids)
        timer.cpu_end("Sliding Window NMS")

        # Submit current batch's post-processing to thread
        pending_postproc = executor.submit(_postproc_and_aggregate, predictions_cpu, metas)
        prev_video_ids = current_video_ids

    # Wait for last batch's post-processing
    timer.cpu_start("Post-processing (CPU)")
    if pending_postproc is not None:
        pending_postproc.result()
    timer.cpu_end("Post-processing (CPU)")

    # Submit NMS for any remaining videos
    timer.cpu_start("Sliding Window NMS")
    if is_sliding_window and cfg.post_processing.nms is not None:
        with result_lock:
            for vid in list(raw_result_dict.keys()):
                if vid not in completed_videos:
                    detections = raw_result_dict.pop(vid)
                    completed_videos.add(vid)
                    future = executor.submit(_nms_one_video, vid, detections, cfg.post_processing)
                    nms_futures.append(future)

    # Collect all NMS results
    for future in nms_futures:
        vid, results = future.result()
        final_result_dict[vid] = results
    timer.cpu_end("Sliding Window NMS")

    executor.shutdown(wait=False)

    # Use NMS'd results if sliding window, otherwise use raw results directly
    if is_sliding_window and cfg.post_processing.nms is not None:
        result_dict = final_result_dict
    else:
        result_dict = raw_result_dict

    # swap back to original model weights
    if model_ema is not None:
        _swap_ema_weights(model, model_ema)

    result_eval = dict(results=result_dict)
    if cfg.post_processing.save_dict:
        result_path = os.path.join(cfg.work_dir, "result_detection.json")
        with open(result_path, "w") as out:
            json.dump(result_eval, out)

    primary_metric = None
    if not not_eval:
        timer.cpu_start("Evaluation")
        evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
        logger.info("Evaluation starts...")
        metrics_dict = evaluator.evaluate()
        evaluator.logging(logger)
        timer.cpu_end("Evaluation")
        if "average_mAP" in metrics_dict:
            primary_metric = metrics_dict["average_mAP"]
        elif "mAP" in metrics_dict:
            primary_metric = metrics_dict["mAP"]

        # Write metrics.json for GUI consumption
        metrics_path = os.path.join(cfg.work_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(_jsonify_metrics(metrics_dict), f, indent=2)

    timer.log_summary(logger, num_batches)

    return primary_metric
