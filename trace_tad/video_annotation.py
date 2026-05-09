"""Render temporal detections back onto source videos."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


Detection = Mapping[str, Any]
cv2: Any | None = None
np: Any | None = None
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class _RenderDetection:
    start: float
    end: float
    label: str
    score: float
    color: tuple[int, int, int]
    text: str


@dataclass
class _TimelineOverlay:
    y0: int
    bar_h: int
    x0: int
    x1: int
    duration: float
    scale: float
    top: int
    layer_pm: Any
    inv_alpha: Any
    scratch: Any

    def apply(self, frame: Any) -> None:
        bottom = self.top + self.layer_pm.shape[0]
        roi = frame[self.top:bottom, :, :]
        np.multiply(roi, self.inv_alpha, out=self.scratch, casting="unsafe")
        np.add(self.scratch, self.layer_pm, out=self.scratch)
        np.clip(self.scratch, 0, 255, out=self.scratch)
        roi[:] = self.scratch


class _ActiveDetectionIndex:
    def __init__(self, detections: list[_RenderDetection]):
        self.detections = detections
        self.by_start = sorted(range(len(detections)), key=lambda idx: detections[idx].start)
        self.by_end = sorted(range(len(detections)), key=lambda idx: detections[idx].end)
        self.next_start = 0
        self.next_end = 0
        self.active_ids: set[int] = set()
        self.last_time = float("-inf")

    def at(self, current_time: float) -> list[_RenderDetection]:
        if current_time < self.last_time:
            self.next_start = 0
            self.next_end = 0
            self.active_ids.clear()
        self.last_time = current_time

        while self.next_start < len(self.by_start):
            idx = self.by_start[self.next_start]
            if self.detections[idx].start > current_time:
                break
            self.active_ids.add(idx)
            self.next_start += 1

        while self.next_end < len(self.by_end):
            idx = self.by_end[self.next_end]
            if self.detections[idx].end >= current_time:
                break
            self.active_ids.discard(idx)
            self.next_end += 1

        return [self.detections[idx] for idx in self.active_ids]


def _format_eta(seconds: float) -> str:
    if seconds < 0 or not seconds < float("inf"):
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _strip_known_video_extension(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return name[:-len(ext)] if ext in VIDEO_EXTENSIONS else name


def _prediction_name_for_video(video_path: str) -> str:
    name = os.path.basename(video_path)
    lower_name = name.lower()
    if lower_name.endswith(".remux.mp4"):
        return _strip_known_video_extension(name[:-len(".remux.mp4")])
    if lower_name.endswith(".h264.mp4"):
        return _strip_known_video_extension(name[:-len(".h264.mp4")])
    return os.path.splitext(name)[0]


def filter_predictions(
    predictions: Mapping[str, list[Detection]],
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    """Return predictions with detections below ``threshold`` removed."""
    threshold = max(0.0, min(1.0, float(threshold)))
    filtered: dict[str, list[dict[str, Any]]] = {}
    for video_name, detections in predictions.items():
        kept = [
            dict(det)
            for det in detections
            if float(det.get("score", 0.0)) >= threshold
        ]
        filtered[video_name] = kept
    return filtered


def render_annotated_videos(
    video_paths: list[str],
    predictions: Mapping[str, list[Detection]],
    output_dir: str,
    threshold: float = 0.0,
    logger: Any | None = None,
) -> dict[str, str]:
    """Write annotated MP4s for ``video_paths`` and return video-name -> path."""
    _require_video_deps()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filtered = filter_predictions(predictions, threshold)
    outputs: dict[str, str] = {}

    for video_path in video_paths:
        video_name = _prediction_name_for_video(video_path)
        detections = filtered.get(video_name, [])
        output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
        try:
            if logger:
                logger.info(
                    f"Rendering annotated video: {video_name} "
                    f"({len(detections)} detections)"
                )
            _render_single_video(
                video_path,
                output_path,
                video_name,
                detections,
                threshold,
                logger=logger,
            )
            outputs[video_name] = output_path
            if logger:
                logger.info(f"Annotated video saved to: {output_path}")
        except Exception as exc:
            if logger:
                logger.warning(f"Could not render annotated video for {video_name}: {exc}")
            else:
                raise

    return outputs


def _require_video_deps() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "Annotated video rendering requires numpy and opencv-python-headless. "
            "Install project requirements before using --annotated-video."
        ) from exc
    cv2 = cv2_module
    np = np_module


def _render_single_video(
    input_path: str,
    output_path: str,
    video_name: str,
    detections: list[Detection],
    threshold: float,
    logger: Any | None = None,
) -> None:
    assert cv2 is not None
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {input_path}")
    if logger:
        logger.info(
            f"Annotated render starts: {video_name} "
            f"({frame_count} frames, {width}x{height})"
        )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    pts = _load_pts(input_path)
    timeline_duration = _timeline_duration(detections, frame_count, fps, pts)
    render_detections = _prepare_render_detections(detections, threshold)
    scale = _frame_scale(width, height)
    timeline = _build_timeline_overlay(width, height, render_detections, timeline_duration, scale)
    active_index = _ActiveDetectionIndex(render_detections)

    frame_index = 0
    render_started_at = time.monotonic()
    last_progress_log_at = 0.0
    last_percent_bucket = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            current_time = _frame_time(frame_index, fps, pts)
            active = active_index.at(current_time)
            annotated = _draw_frame(
                frame,
                video_name,
                render_detections,
                active,
                current_time,
                threshold,
                timeline,
                scale,
            )
            writer.write(annotated)
            frame_index += 1
            if logger and frame_count > 0:
                now = time.monotonic()
                percent = min(100.0, frame_index * 100.0 / frame_count)
                percent_bucket = int(percent)
                should_log = (
                    frame_index == 1
                    or frame_index == frame_count
                    or percent_bucket > last_percent_bucket
                    or now - last_progress_log_at >= 5.0
                )
                if not should_log:
                    continue
                elapsed = max(0.001, now - render_started_at)
                render_fps = frame_index / elapsed
                remaining = (frame_count - frame_index) / render_fps if render_fps > 0 else float("inf")
                logger.info(
                    f"Annotated render progress: {video_name} "
                    f"{frame_index}/{frame_count} frames "
                    f"({percent:.1f}%, {render_fps:.1f} fps, eta {_format_eta(remaining)})"
                )
                last_progress_log_at = now
                last_percent_bucket = percent_bucket
    finally:
        capture.release()
        writer.release()
    if logger:
        logger.info(
            f"Annotated render complete: {video_name} "
            f"({frame_index}/{frame_count} frames)"
        )


def _load_pts(video_path: str) -> Any | None:
    assert np is not None
    pts_path = f"{video_path}.pts.npy"
    if not os.path.isfile(pts_path):
        return None
    try:
        pts = np.load(pts_path)
    except Exception:
        return None
    if pts.ndim != 1 or len(pts) == 0:
        return None
    return pts.astype(np.float64)


def _frame_time(frame_index: int, fps: float, pts: Any | None) -> float:
    if pts is not None and frame_index < len(pts):
        return float(pts[frame_index] - pts[0])
    return frame_index / fps


def _timeline_duration(
    detections: list[Detection],
    frame_count: int,
    fps: float,
    pts: Any | None,
) -> float:
    if pts is not None and len(pts) > 1:
        duration = float(pts[-1] - pts[0]) + (1.0 / fps)
    else:
        duration = frame_count / fps if frame_count > 0 else 0.0
    if detections:
        duration = max(duration, max(_segment_end(det) for det in detections))
    return max(duration, 0.001)


def _segment_start(det: Detection) -> float:
    return float(det.get("segment", [0.0, 0.0])[0])


def _segment_end(det: Detection) -> float:
    return float(det.get("segment", [0.0, 0.0])[1])


def _label_color(label: str) -> tuple[int, int, int]:
    assert cv2 is not None and np is not None
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    hue = int(digest[0]) % 180
    sat = 120 + int(digest[1]) % 80
    val = 190 + int(digest[2]) % 45
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _prepare_render_detections(
    detections: list[Detection],
    threshold: float,
) -> list[_RenderDetection]:
    prepared: list[_RenderDetection] = []
    for det in detections:
        score = float(det.get("score", 0.0))
        if score < threshold:
            continue
        label = str(det.get("label", "unknown"))
        color = _label_color(label)
        prepared.append(
            _RenderDetection(
                start=_segment_start(det),
                end=_segment_end(det),
                label=label,
                score=score,
                color=color,
                text=f"{label}  {score:.2f}",
            )
        )
    prepared.sort(key=lambda det: (det.start, det.end, -det.score))
    return prepared


def _frame_scale(width: int, height: int) -> float:
    return max(0.7, min(width, height) / 900.0)


def _draw_frame(
    frame: Any,
    video_name: str,
    detections: list[_RenderDetection],
    active: list[_RenderDetection],
    current_time: float,
    threshold: float,
    timeline: _TimelineOverlay,
    scale: float,
) -> Any:
    out = frame
    height, width = out.shape[:2]
    pad = int(14 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX

    _blend_rect(out, (0, 0), (width, int(86 * scale)), (12, 14, 18), 0.58)

    title = video_name
    cv2.putText(out, title, (pad, int(30 * scale)), font, 0.62 * scale, (245, 245, 240), max(1, int(2 * scale)), cv2.LINE_AA)
    subtitle = f"{_fmt_time(current_time)}  |  threshold {threshold:.2f}  |  {len(detections)} detections"
    cv2.putText(out, subtitle, (pad, int(58 * scale)), font, 0.46 * scale, (196, 202, 210), max(1, int(1 * scale)), cv2.LINE_AA)

    if active:
        y = int(104 * scale)
        for det in sorted(active, key=lambda d: d.score, reverse=True)[:3]:
            text = det.text
            (tw, th), _ = cv2.getTextSize(text, font, 0.58 * scale, max(1, int(2 * scale)))
            box_w = min(width - pad * 2, tw + int(28 * scale))
            box_h = int(34 * scale)
            _rounded_rect(out, (pad, y), (pad + box_w, y + box_h), (16, 18, 22), radius=int(9 * scale), alpha=0.74)
            cv2.circle(out, (pad + int(14 * scale), y + box_h // 2), int(5 * scale), det.color, -1, cv2.LINE_AA)
            cv2.putText(out, text, (pad + int(26 * scale), y + int(23 * scale)), font, 0.58 * scale, (248, 248, 244), max(1, int(2 * scale)), cv2.LINE_AA)
            y += int(42 * scale)
    else:
        idle = "No active behavior"
        cv2.putText(out, idle, (pad, int(116 * scale)), font, 0.54 * scale, (202, 207, 214), max(1, int(1 * scale)), cv2.LINE_AA)

    timeline.apply(out)
    _draw_timeline_playhead(out, current_time, timeline)
    return out


def _build_timeline_overlay(
    width: int,
    height: int,
    detections: list[_RenderDetection],
    duration: float,
    scale: float,
) -> _TimelineOverlay:
    margin = int(22 * scale)
    bar_h = min(max(1, height), max(16, int(20 * scale)))
    y0 = max(0, height - margin - bar_h)
    x0 = min(margin, max(0, width - 1))
    x1 = min(width, max(x0 + 1, width - margin))
    top = max(0, y0 - int(8 * scale))
    bottom = min(height, y0 + bar_h + int(8 * scale) + 1)

    layer_pm = np.zeros((bottom - top, width, 3), dtype=np.float32)
    alpha = np.zeros((bottom - top, width), dtype=np.float32)
    local_y0 = y0 - top
    _paint_rounded_rect_pm(
        layer_pm,
        alpha,
        (x0, local_y0),
        (x1, local_y0 + bar_h),
        (20, 22, 26),
        radius=int(8 * scale),
        alpha=0.72,
    )
    for det in detections:
        start = max(0.0, min(duration, det.start))
        end = max(start, min(duration, det.end))
        sx = int(x0 + (x1 - x0) * start / duration)
        ex = max(sx + 2, int(x0 + (x1 - x0) * end / duration))
        alpha_value = 0.45 + min(0.4, det.score * 0.4)
        _paint_rounded_rect_pm(
            layer_pm,
            alpha,
            (sx, local_y0 + 3),
            (ex, local_y0 + bar_h - 3),
            det.color,
            radius=int(5 * scale),
            alpha=alpha_value,
        )

    inv_alpha = (1.0 - alpha[:, :, None]).astype(np.float32)
    scratch = np.empty_like(layer_pm)
    return _TimelineOverlay(
        y0=y0,
        bar_h=bar_h,
        x0=x0,
        x1=x1,
        duration=duration,
        scale=scale,
        top=top,
        layer_pm=layer_pm,
        inv_alpha=inv_alpha,
        scratch=scratch,
    )


def _draw_timeline_playhead(
    frame: Any,
    current_time: float,
    timeline: _TimelineOverlay,
) -> None:
    px = int(
        timeline.x0
        + (timeline.x1 - timeline.x0)
        * max(0.0, min(timeline.duration, current_time))
        / timeline.duration
    )
    cv2.line(
        frame,
        (px, timeline.y0 - int(7 * timeline.scale)),
        (px, timeline.y0 + timeline.bar_h + int(7 * timeline.scale)),
        (245, 245, 240),
        max(1, int(2 * timeline.scale)),
        cv2.LINE_AA,
    )


def _draw_timeline(
    frame: Any,
    detections: list[Detection],
    current_time: float,
    duration: float,
    palette: Mapping[str, tuple[int, int, int]],
    scale: float,
) -> None:
    height, width = frame.shape[:2]
    margin = int(22 * scale)
    bar_h = max(16, int(20 * scale))
    y0 = height - margin - bar_h
    x0 = margin
    x1 = width - margin

    _rounded_rect(frame, (x0, y0), (x1, y0 + bar_h), (20, 22, 26), radius=int(8 * scale), alpha=0.72)
    for det in detections:
        start = max(0.0, min(duration, _segment_start(det)))
        end = max(start, min(duration, _segment_end(det)))
        sx = int(x0 + (x1 - x0) * start / duration)
        ex = max(sx + 2, int(x0 + (x1 - x0) * end / duration))
        color = palette.get(str(det.get("label", "")), (90, 190, 255))
        alpha = 0.45 + min(0.4, float(det.get("score", 0.0)) * 0.4)
        _rounded_rect(frame, (sx, y0 + 3), (ex, y0 + bar_h - 3), color, radius=int(5 * scale), alpha=alpha)

    px = int(x0 + (x1 - x0) * max(0.0, min(duration, current_time)) / duration)
    cv2.line(frame, (px, y0 - int(7 * scale)), (px, y0 + bar_h + int(7 * scale)), (245, 245, 240), max(1, int(2 * scale)), cv2.LINE_AA)


def _clip_box(
    image: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    x0 = max(0, min(width, int(top_left[0])))
    y0 = max(0, min(height, int(top_left[1])))
    x1 = max(0, min(width, int(bottom_right[0])))
    y1 = max(0, min(height, int(bottom_right[1])))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _rounded_rect_mask(width: int, height: int, radius: int) -> Any:
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = max(0, min(radius, width // 2, height // 2))
    if radius == 0:
        cv2.rectangle(mask, (0, 0), (width, height), 255, -1)
        return mask
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (width - radius - 1, radius), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (radius, height - radius - 1), radius, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (width - radius - 1, height - radius - 1), radius, 255, -1, cv2.LINE_AA)
    return mask


def _blend_rect(
    image: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    clipped = _clip_box(image, top_left, bottom_right)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    roi = image[y0:y1, x0:x1, :]
    color_arr = np.asarray(color, dtype=np.float32)
    blended = roi.astype(np.float32) * (1.0 - alpha) + color_arr * alpha
    roi[:] = blended


def _paint_rounded_rect_pm(
    layer_pm: Any,
    alpha_layer: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    alpha: float,
) -> None:
    clipped = _clip_box(layer_pm, top_left, bottom_right)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    rect_w = x1 - x0
    rect_h = y1 - y0
    mask = _rounded_rect_mask(rect_w, rect_h, radius).astype(np.float32) / 255.0
    src_alpha = mask * max(0.0, min(1.0, alpha))
    dst_alpha = alpha_layer[y0:y1, x0:x1]
    inv_src_alpha = 1.0 - src_alpha

    layer_roi = layer_pm[y0:y1, x0:x1, :]
    color_arr = np.asarray(color, dtype=np.float32)
    layer_roi[:] = color_arr * src_alpha[:, :, None] + layer_roi * inv_src_alpha[:, :, None]
    dst_alpha[:] = src_alpha + dst_alpha * inv_src_alpha


def _rounded_rect(
    image: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    alpha: float,
) -> None:
    clipped = _clip_box(image, top_left, bottom_right)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    roi = image[y0:y1, x0:x1, :]
    rect_w = x1 - x0
    rect_h = y1 - y0
    mask = _rounded_rect_mask(rect_w, rect_h, radius).astype(np.float32) / 255.0
    alpha_mask = mask * max(0.0, min(1.0, alpha))
    color_arr = np.asarray(color, dtype=np.float32)
    blended = roi.astype(np.float32) * (1.0 - alpha_mask[:, :, None]) + color_arr * alpha_mask[:, :, None]
    roi[:] = blended


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"
