"""
Pure decord + torchvision replacements for mmaction video pipeline steps.
Replaces: mmaction.DecordInit, DecordDecode, Resize, RandomResizedCrop,
          CenterCrop, Flip, ImgAug, ColorJitter, FormatShape.
"""
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import cv2
import decord
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from ..builder import PIPELINES


class _VideoReaderCache:
    """Per-process LRU cache of decord.VideoReader instances.

    Virtual clips read frame ranges directly from the original source video
    rather than from pre-extracted clip files. Without caching, every
    __getitem__ call would reopen the source (parsing moov atom etc.). With
    caching + DataLoader's persistent_workers=True, hot source videos stay
    open across samples and across epochs.

    Each PyTorch worker is a separate process, so each gets its own cache.
    """

    def __init__(self, maxsize: int = 8):
        self._cache: "OrderedDict[tuple[str, int, int, int], decord.VideoReader]" = OrderedDict()
        self._max = maxsize

    def get(self, path: str, num_threads: int, width: int = -1, height: int = -1) -> decord.VideoReader:
        path = os.path.abspath(path)
        key = (path, int(width), int(height), int(num_threads))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= self._max:
            _, old = self._cache.popitem(last=False)
            del old
        vr = decord.VideoReader(path, width=width, height=height, num_threads=num_threads)
        self._cache[key] = vr
        return vr


_video_reader_cache = _VideoReaderCache()


@PIPELINES.register_module()
class VideoInit:
    """Open the video for this sample.

    For virtual clips (sample dict contains `clip_frame_count`), `total_frames`
    is the clip's logical length, not the source video's full length, so that
    downstream LoadFrames/LoadSnippetFrames sample within the clip's span. The
    actual seek to source-frame coordinates happens in VideoDecode by adding
    `source_frame_offset` to the clip-local indices.
    """

    def __init__(self, num_threads: int = 4, resize=None, width: int = -1, height: int = -1):
        self.num_threads = num_threads
        if resize is not None:
            if isinstance(resize, int):
                height = width = resize
            else:
                height, width = resize
        self.width = int(width)
        self.height = int(height)

    def __call__(self, results):
        filename = results["filename"]
        vr = _video_reader_cache.get(filename, self.num_threads, width=self.width, height=self.height)
        results["video_reader"] = vr
        if "clip_frame_count" in results:
            results["total_frames"] = int(results["clip_frame_count"])
        else:
            results["total_frames"] = len(vr)
        results["avg_fps"] = vr.get_avg_fps()
        return results


@PIPELINES.register_module()
class VideoTemporalAugment:
    """Random temporal speed augmentation applied after LoadFrames, before VideoDecode.

    Resamples frame_inds to simulate playback speed changes, and scales
    gt_segments proportionally.

    Args:
        speed_range: [min_speed, max_speed], e.g. [0.8, 1.2].
        p: Probability of applying the augmentation.
    """

    def __init__(self, speed_range=(0.8, 1.2), p=0.5):
        self.speed_range = speed_range
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results

        frame_inds = results.get("frame_inds")
        if frame_inds is None or len(frame_inds) == 0:
            return results

        speed = random.uniform(*self.speed_range)
        if abs(speed - 1.0) < 1e-6:
            return results

        total_frames = results.get("total_frames", None)
        orig_inds = frame_inds.flatten().astype(np.float64)
        num_frames = len(orig_inds)

        # Resample frame indices around the center at the new speed.
        # speed > 1 → cover a wider source range (faster playback)
        # speed < 1 → cover a narrower source range (slower playback)
        center = (orig_inds[0] + orig_inds[-1]) / 2.0
        half_span = (orig_inds[-1] - orig_inds[0]) / 2.0
        new_half = half_span * speed
        new_start = center - new_half
        new_end = center + new_half

        new_inds = np.linspace(new_start, new_end, num_frames)

        # Clamp to valid frame range
        max_idx = (total_frames - 1) if total_frames is not None else orig_inds.max()
        new_inds = np.clip(new_inds, 0, max_idx)
        new_inds = np.round(new_inds).astype(frame_inds.dtype)

        results["frame_inds"] = new_inds.reshape(frame_inds.shape)

        # Scale gt_segments by inverse speed factor (1/speed)
        if "gt_segments" in results and results["gt_segments"] is not None:
            gt_segments = results["gt_segments"].astype(np.float64)
            # The temporal origin shifts and scales. Transform segment
            # boundaries relative to the original frame range.
            orig_start = orig_inds[0]
            orig_span = orig_inds[-1] - orig_inds[0]
            if orig_span > 0:
                # Normalize segments to [0,1] in the original span, then
                # map to the new span.  Since the *content* that was at
                # position t now appears at position t/speed, we scale
                # by 1/speed relative to the temporal center.
                seg_center = (gt_segments[:, 0] + gt_segments[:, 1]) / 2.0
                seg_half = (gt_segments[:, 1] - gt_segments[:, 0]) / 2.0

                # Scale segment duration and position around the window center
                window_center = orig_span / 2.0
                new_seg_center = window_center + (seg_center - window_center) / speed
                new_seg_half = seg_half / speed

                gt_segments[:, 0] = new_seg_center - new_seg_half
                gt_segments[:, 1] = new_seg_center + new_seg_half

                # Clamp to the output temporal range
                output_len = num_frames  # after resampling, temporal length is preserved
                gt_segments = np.clip(gt_segments, 0, output_len)

            results["gt_segments"] = gt_segments.astype(np.float32)

        return results


@PIPELINES.register_module()
class VideoDecode:
    """Decode the chosen frames into image arrays.

    For virtual clips, `frame_inds` are clip-local; `source_frame_offset`
    translates them into source-video coordinates before `get_batch`. The
    VideoReader is removed from `results` (workers don't need to ship it
    downstream) but kept alive in the per-process cache.
    """

    def __call__(self, results):
        frame_inds = results["frame_inds"]
        vr = results["video_reader"]
        flat_inds = frame_inds.flatten()
        offset = int(results.get("decode_frame_offset", results.get("source_frame_offset", 0)))
        if offset:
            flat_inds = flat_inds + offset
        # clamp to valid source-frame range
        flat_inds = np.clip(flat_inds, 0, len(vr) - 1)
        imgs = vr.get_batch(flat_inds.tolist()).asnumpy()  # [N, H, W, 3]
        imgs = imgs.reshape(*frame_inds.shape, *imgs.shape[1:])  # [..., H, W, 3]
        results["imgs"] = list(imgs) if imgs.ndim > 3 else [imgs]
        del results["video_reader"]
        return results


def _resize_img(img, scale):
    """Resize image. scale=(-1, 256) means shorter side to 256."""
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        w, h = img.size

    if isinstance(scale, int):
        short, long = min(h, w), max(h, w)
        new_short = scale
        new_long = int(long * new_short / short)
        new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    elif scale[0] == -1:
        short, long = min(h, w), max(h, w)
        new_short = scale[1]
        new_long = int(long * new_short / short)
        new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    elif scale[1] == -1:
        short, long = min(h, w), max(h, w)
        new_short = scale[0]
        new_long = int(long * new_short / short)
        new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    else:
        new_h, new_w = scale[0], scale[1]

    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        return np.array(pil)
    else:
        return TF.resize(img, (new_h, new_w))


@PIPELINES.register_module()
class VideoResize:
    """Replaces mmaction.Resize. scale=(-1, 256) resizes shorter side to 256."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, results):
        imgs = results["imgs"]
        results["imgs"] = [_resize_img(img, self.scale) for img in imgs]
        return results


@PIPELINES.register_module()
class VideoBatchResize:
    """Batched resize using cv2 — directly resizes all frames to (H, W)
    without preserving aspect ratio. Much faster than per-frame PIL conversion.

    Args:
        scale: Target (height, width) tuple, e.g. (224, 224).
        interpolation: cv2 interpolation flag. Default INTER_LINEAR.
    """

    def __init__(self, scale, interpolation=cv2.INTER_LINEAR):
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = tuple(scale)
        self.interpolation = interpolation

    def __call__(self, results):
        imgs = results["imgs"]
        th, tw = self.scale

        # fast path: if already the right size, skip
        if isinstance(imgs[0], np.ndarray) and imgs[0].shape[0] == th and imgs[0].shape[1] == tw:
            return results

        results["imgs"] = [
            cv2.resize(img, (tw, th), interpolation=self.interpolation)
            for img in imgs
        ]
        return results


@PIPELINES.register_module()
class VideoRandomResizedCrop:
    """Replaces mmaction.RandomResizedCrop.
    Applies the same random crop to all frames in the clip.
    """

    def __init__(self, area_range=(0.08, 1.0), aspect_ratio_range=(3 / 4, 4 / 3)):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, results):
        imgs = results["imgs"]
        if not imgs:
            return results

        if isinstance(imgs[0], np.ndarray):
            h, w = imgs[0].shape[:2]
        else:
            w, h = imgs[0].size

        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.area_range) * area
            ar = random.uniform(*self.aspect_ratio_range)
            new_w = int(round((target_area * ar) ** 0.5))
            new_h = int(round((target_area / ar) ** 0.5))
            if new_w <= w and new_h <= h:
                x = random.randint(0, w - new_w)
                y = random.randint(0, h - new_h)
                results["imgs"] = [
                    (img[y:y+new_h, x:x+new_w] if isinstance(img, np.ndarray)
                     else TF.crop(img, y, x, new_h, new_w))
                    for img in imgs
                ]
                return results

        # Fallback: center crop
        x = (w - min(w, h)) // 2
        y = (h - min(w, h)) // 2
        s = min(w, h)
        results["imgs"] = [
            (img[y:y+s, x:x+s] if isinstance(img, np.ndarray)
             else TF.center_crop(img, s))
            for img in imgs
        ]
        return results


@PIPELINES.register_module()
class VideoCenterCrop:
    """Replaces mmaction.CenterCrop."""

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, results):
        imgs = results["imgs"]
        ch, cw = self.crop_size

        def _crop(img):
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                y = (h - ch) // 2
                x = (w - cw) // 2
                return img[y:y+ch, x:x+cw]
            else:
                return TF.center_crop(img, self.crop_size)

        results["imgs"] = [_crop(img) for img in imgs]
        return results


@PIPELINES.register_module()
class VideoFlip:
    """Replaces mmaction.Flip. Applies uniform random flip to all frames."""

    def __init__(self, flip_ratio: float = 0.5, direction: str = "horizontal"):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        if random.random() < self.flip_ratio:
            imgs = results["imgs"]
            if self.direction == "horizontal":
                results["imgs"] = [
                    (img[:, ::-1].copy() if isinstance(img, np.ndarray)
                     else TF.hflip(img))
                    for img in imgs
                ]
        return results


@PIPELINES.register_module()
class VideoColorJitter:
    """Applies consistent color jitter across all frames using numpy batch ops.
    Avoids per-frame numpy->PIL->numpy conversion overhead.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast)
        self.saturation = self._check_input(saturation)
        self.hue = self._check_input(hue, center=0, bound=0.5, clip_first_on_zero=False)

    @staticmethod
    def _check_input(value, center=1, bound=float("inf"), clip_first_on_zero=True):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"Value {value} must be non-negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        return value

    def __call__(self, results):
        imgs = results["imgs"]
        if not imgs:
            return results

        # sample parameters once for all frames
        brightness_factor = random.uniform(*self.brightness) if self.brightness else None
        contrast_factor = random.uniform(*self.contrast) if self.contrast else None
        saturation_factor = random.uniform(*self.saturation) if self.saturation else None
        hue_factor = random.uniform(*self.hue) if self.hue else None

        # random order
        fn_idx = list(range(4))
        random.shuffle(fn_idx)

        new_imgs = []
        for img in imgs:
            is_numpy = isinstance(img, np.ndarray)
            if not is_numpy:
                img = np.array(img)
            img = img.astype(np.float32)

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = img * brightness_factor
                elif fn_id == 1 and contrast_factor is not None:
                    mean = img.mean(axis=(0, 1), keepdims=True)
                    img = (img - mean) * contrast_factor + mean
                elif fn_id == 2 and saturation_factor is not None:
                    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                    gray = gray[..., np.newaxis]
                    img = (img - gray) * saturation_factor + gray
                elif fn_id == 3 and hue_factor is not None and hue_factor != 0:
                    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[..., 0] = (hsv[..., 0] + hue_factor * 180) % 180
                    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

            new_imgs.append(np.clip(img, 0, 255).astype(np.uint8))

        results["imgs"] = new_imgs
        return results


@PIPELINES.register_module()
class VideoImgAug:
    """Applies random Gaussian blur to a clip with 50% probability.

    Uses torchvision instead of the deprecated imgaug library.
    """

    def __init__(self, transforms=None, p=0.5, sigma=(0.1, 3.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, results):
        if random.random() > self.p:
            return results
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # kernel size must be odd and large enough for the sigma
        kernel_size = int(2 * round(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 3)
        blur = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        results["imgs"] = [
            np.array(blur(Image.fromarray(img if isinstance(img, np.ndarray) else np.array(img))))
            for img in results["imgs"]
        ]
        return results


@PIPELINES.register_module()
class VideoFormatShape:
    """Replaces mmaction.FormatShape.
    Converts list of [H,W,3] numpy arrays -> [N, 3, T, H, W] tensor (NCTHW).
    """

    def __init__(self, input_format: str = "NCTHW"):
        self.input_format = input_format

    def __call__(self, results):
        imgs = results["imgs"]
        if isinstance(imgs[0], np.ndarray):
            imgs = np.stack(imgs, axis=0)  # [T, H, W, 3]
            imgs = imgs.transpose(3, 0, 1, 2)  # [3, T, H, W]
            imgs = imgs[np.newaxis]  # [1, 3, T, H, W]
        elif isinstance(imgs[0], torch.Tensor):
            imgs = torch.stack(imgs, dim=0)  # [T, C, H, W]
            imgs = imgs.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        results["imgs"] = imgs
        return results


@PIPELINES.register_module()
class VideoNormalize:
    """Pixel normalisation transform (replaces ActionDataPreprocessor).
    Applied BEFORE or as part of the pipeline if not done in BackboneWrapper.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        imgs = results["imgs"]
        if isinstance(imgs, np.ndarray):
            results["imgs"] = (imgs.astype(np.float32) - self.mean) / self.std
        elif isinstance(imgs, torch.Tensor):
            mean = torch.tensor(self.mean, device=imgs.device).reshape(1, 3, 1, 1, 1)
            std = torch.tensor(self.std, device=imgs.device).reshape(1, 3, 1, 1, 1)
            results["imgs"] = (imgs.float() - mean) / std
        return results
