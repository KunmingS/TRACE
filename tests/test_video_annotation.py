import numpy as np

from trace_tad import video_annotation as va


class Cv2Stub:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def rectangle(self, img, pt1, pt2, color, thickness):
        x0, y0 = pt1
        x1, y1 = pt2
        img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)] = color

    def circle(self, img, center, radius, color, thickness, lineType=None):
        cx, cy = center
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = color

    def line(self, img, pt1, pt2, color, thickness, lineType=None):
        x0, y0 = pt1
        _, y1 = pt2
        x0 = max(0, min(img.shape[1] - 1, x0))
        y0, y1 = sorted((y0, y1))
        y0 = max(0, min(img.shape[0] - 1, y0))
        y1 = max(0, min(img.shape[0] - 1, y1))
        img[y0:y1 + 1, max(0, x0 - thickness):min(img.shape[1], x0 + thickness + 1)] = color

    def putText(self, *args, **kwargs):
        return None

    def getTextSize(self, text, font, scale, thickness):
        return (int(len(text) * 8 * scale), int(16 * scale)), 0


def _install_video_stubs(monkeypatch):
    monkeypatch.setattr(va, "np", np)
    monkeypatch.setattr(va, "cv2", Cv2Stub())


def test_active_detection_index_tracks_monotonic_time_and_resets(monkeypatch):
    _install_video_stubs(monkeypatch)
    detections = [
        va._RenderDetection(0.0, 1.0, "walk", 0.9, (10, 20, 30), "walk  0.90"),
        va._RenderDetection(2.0, 3.0, "rear", 0.6, (40, 50, 60), "rear  0.60"),
    ]
    index = va._ActiveDetectionIndex(detections)

    assert [det.label for det in index.at(0.5)] == ["walk"]
    assert index.at(1.5) == []
    assert [det.label for det in index.at(2.5)] == ["rear"]
    assert [det.label for det in index.at(0.5)] == ["walk"]


def test_draw_frame_uses_in_place_optimized_overlay(monkeypatch):
    _install_video_stubs(monkeypatch)
    detections = [
        va._RenderDetection(0.0, 1.0, "walk", 0.9, (10, 20, 30), "walk  0.90"),
    ]
    frame = np.full((120, 200, 3), 128, dtype=np.uint8)
    scale = va._frame_scale(200, 120)
    timeline = va._build_timeline_overlay(200, 120, detections, 4.0, scale)

    out = va._draw_frame(
        frame,
        "trial",
        detections,
        detections,
        current_time=0.5,
        threshold=0.5,
        timeline=timeline,
        scale=scale,
    )

    assert out is frame
    assert out.shape == (120, 200, 3)
    assert int(out.sum()) != 120 * 200 * 3 * 128
