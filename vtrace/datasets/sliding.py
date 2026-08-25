import json
import math
import os
import numpy as np
class Compose:
    # Defined as a top-level class (not a closure) so dataset instances remain
    # picklable for DataLoader workers under spawn/forkserver start methods
    # (Python 3.14 default on Linux, always on macOS/Windows).
    def __init__(self, pipeline):
        from .builder import PIPELINES
        self.transforms = [PIPELINES.build(t) if isinstance(t, dict) else t for t in pipeline]

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
        return results

from .builder import DATASETS, get_class_index


@DATASETS.register_module()
class SlidingWindowDataset:
    def __init__(
        self,
        ann_file,  # path of the annotation json file
        subset_name,  # name of the subset, such as training, validation, testing
        data_path,  # folder path of the raw video / pre-extracted feature
        pipeline,  # data pipeline
        class_map,  # path of the class map, convert the class id to category name
        filter_gt=False,  # if True, filter out those gt has the scale smaller than 0.01
        class_agnostic=False,  # if True, the class index will be replaced by 0
        block_list=None,  # some videos might be missed in the features or videos, we need to block them
        test_mode=False,  # if True, running on test mode with no annotation
        # for feature setting
        feature_stride=-1,  # the frames between two adjacent features, such as 4 frames
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        offset_frames=0,  # the start offset frame of the input feature
        # for sliding window setting
        window_size=-1,  # the number of features in a window
        window_overlap_ratio=0.25,  # the overlap ratio of two adjacent windows
        ioa_thresh=0.75,  # gt completeness threshold, window_gt_mode="complete" only
        window_gt_mode="truncate",  # "truncate" | "complete"; see classify_window
        base_jitter=0.0,  # per-read window shift, as a fraction of window_size (0 = off)
        fps=-1,  # some annotations are based on video-seconds
        keep_empty_gt=False,  # if True, keep videos/clips with NO gt as background (negative) windows
        logger=None,
    ):
        super(SlidingWindowDataset, self).__init__()

        # basic settings
        self.data_path = data_path
        self.block_list = block_list
        self.ann_file = ann_file
        self.subset_name = subset_name
        self.logger = logger.info if logger != None else print
        self.class_map = self.get_class_map(class_map)
        self.class_agnostic = class_agnostic
        self.filter_gt = filter_gt
        self.test_mode = test_mode
        self.keep_empty_gt = keep_empty_gt
        self.pipeline = Compose(pipeline)

        # feature settings
        self.feature_stride = int(feature_stride)
        self.sample_stride = int(sample_stride)
        self.offset_frames = int(offset_frames)
        self.snippet_stride = int(feature_stride * sample_stride)
        self.fps = fps

        # window settings
        self.window_size = int(window_size)
        self.window_stride = int(window_size * (1 - window_overlap_ratio))
        self.ioa_thresh = ioa_thresh
        if window_gt_mode not in ("truncate", "complete"):
            raise ValueError(
                f"window_gt_mode must be 'truncate' or 'complete', got {window_gt_mode!r}"
            )
        self.window_gt_mode = window_gt_mode
        # Eval must be deterministic and must slide the fixed grid, so jitter is a
        # train-side setting only. Forced off under test_mode rather than trusted to
        # the config, since test_mode is what distinguishes the eval datasets.
        self.base_jitter = 0.0 if test_mode else max(0.0, float(base_jitter))

        self.get_dataset()
        self.logger(
            f"{self.subset_name} subset: {len(set([data[0] for data in self.data_list]))} videos, "
            f"truncated as {len(self.data_list)} windows."
        )

    def get_dataset(self):
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list != None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list, "r") as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []

        self.data_list = []
        for video_name, video_info in anno_database.items():
            if (video_name in blocked_videos) or (video_info["subset"] not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno == None:  # have no valid gt
                    if not self.keep_empty_gt:
                        continue
                    # keep as a background (negative) clip: empty gt -> falls to the
                    # unconditional else-branch in split_video_to_windows below
                    video_anno = empty_annotation()

            tmp_data_list = self.split_video_to_windows(video_name, video_info, video_anno)
            self.data_list.extend(tmp_data_list)
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def split_video_to_windows(self, video_name, video_info, video_anno):
        # need: video frame, video duration, video fps
        if self.fps > 0:
            num_frames = int(video_info["duration"] * self.fps)
        else:
            num_frames = video_info["frame"]

        video_snippet_centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False  # whether it is the last window

        # ceil, not floor: with floor, the loop stopped before the index whose window
        # runs past the end, so the clamp below was never reached and the last
        # `snippet_num % window_size` snippets of every video were dropped (a 21364-frame
        # video at overlap0 lost 628 frames / 21 s). Only overlap0 was affected --- with
        # any overlap the next start still lands inside the video --- and it hit eval too,
        # where val/test slide at overlap0. The `last_window` break keeps the extra index
        # from producing a duplicate window when the length divides evenly.
        for idx in range(max(1, math.ceil(snippet_num / self.window_stride))):
            window_start = idx * self.window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:  # this is the last window
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]

            window_anno = self.window_annotation(video_anno, window_snippet_centers)
            if window_anno is not None:
                entry = [video_name, video_info, window_anno, window_snippet_centers]
                jitter = self._jitter_record(
                    window_start, max(0, snippet_num - self.window_size),
                    video_snippet_centers, video_anno,
                )
                if jitter is not None:
                    entry.append(jitter)
                data_list.append(entry)

            if last_window:  # the last window
                break

        return data_list

    def window_annotation(self, video_anno, window, gt_mode=None):
        """GT for one window, or None to drop the window.

        The single place that maps (a video's full GT, a window) -> that window's GT, so
        the build path and the jittered read path cannot drift apart. ``gt_mode``
        overrides ``self.window_gt_mode`` for one call (rare-class anchors always
        truncate).

        A GT-free annotation — test_mode's ``{}``, or a ``keep_empty_gt`` background clip
        — is passed through untouched: sending it through the filter would reject it and
        silently drop every background window.
        """
        if (video_anno == {}) or (len(video_anno["gt_segments"]) == 0):
            return video_anno
        return self.classify_window(video_anno, window[0], window[-1], gt_mode=gt_mode)

    def _jitter_record(self, window_start, max_start, video_snippet_centers, video_anno, gt_mode=None):
        """Bounds a window may be re-drawn within on each read, or None for a fixed window.

        Returned as the entry's 5th field and consumed by
        ``PlainSlidingDataset.__getitem__``. Rare-class anchors build their own record
        with the same shape but a narrower range (one phase stratum) and
        ``gt_mode="truncate"``.
        """
        if self.base_jitter <= 0:
            return None
        radius = max(1, int(self.base_jitter * self.window_size))
        lo = max(0, window_start - radius)
        hi = min(max_start, window_start + radius)
        if lo >= hi:  # no room to move (video barely longer than one window)
            return None
        return (lo, hi, video_snippet_centers, video_anno, gt_mode)

    def classify_window(self, video_anno, window_start_frame, window_end_frame, gt_mode=None):
        """Decide what a window of a GT-bearing video contributes to the train set.

        ``window_gt_mode="truncate"`` (default) — every GT overlapping the window, clipped
        to it. A window overlapping no GT is background: kept with ``keep_empty_gt``,
        dropped otherwise. Correct for the per-frame classifier that is now the whole
        model: a partially visible bout has its visible frames labeled positive and the
        rest of the window labeled background, which is exactly the truth.

        ``window_gt_mode="complete"`` — the historical rule: only GTs at least
        ``ioa_thresh`` complete inside the window survive, and a window that merely clips
        a GT is dropped as ambiguous. This is a *segment*-completeness rule inherited from
        the localization head, which no longer exists. It also mislabels: a bout clipped
        by a window edge stays on screen but is excluded from the GT, so the per-frame
        head is trained to call those frames background, which can silently discard a
        sizeable share of the labeled frames. Kept only to reproduce runs from before
        the switch.

        The background case only shows up once a video is LONGER than the window: a
        clip cache cut to window_size has one window, which either contains its GT or
        has none at all. On an hours-long video it is nearly every window, and dropping
        it left needle-in-background corpora training (and evaluating) without any
        empty video at all. ``keep_empty_gt`` defaults to False, so every config that
        does not opt in keeps its historical window set exactly.
        """
        gt_segments = video_anno["gt_segments"]
        gt_labels = video_anno["gt_labels"]
        anchor = np.array([window_start_frame, window_end_frame])

        # truncate the gt segments inside the window and compute the completeness
        gt_completeness, truncated_gt = compute_gt_completeness(gt_segments, anchor)

        if (gt_mode or self.window_gt_mode) == "truncate":
            truncated = self.truncate_window_gt(video_anno, window_start_frame, window_end_frame)
            if truncated is not None:
                return truncated
            return empty_annotation() if self.keep_empty_gt else None

        valid_idx = gt_completeness > self.ioa_thresh
        if np.sum(valid_idx) > 0:
            return dict(gt_segments=truncated_gt[valid_idx], gt_labels=gt_labels[valid_idx])
        if self.keep_empty_gt and not np.any(gt_completeness > 0):
            return empty_annotation()
        return None

    def truncate_window_gt(self, video_anno, window_start_frame, window_end_frame):
        """Every GT overlapping the window, truncated to it — no completeness filter.

        For a dense per-frame classifier a partially-visible bout is still correctly
        labeled (its visible frames are positive, the rest of the window is background),
        so ``ioa_thresh`` — a segment-completeness rule left over from the removed
        localization head — must not apply. Backs ``window_gt_mode="truncate"`` and the
        rare-class anchors, whose entering/leaving windows are the whole point.
        Returns None if nothing overlaps.
        """
        gt_completeness, truncated_gt = compute_gt_completeness(video_anno["gt_segments"], np.array([window_start_frame, window_end_frame]))
        overlap_idx = gt_completeness > 0
        if not np.any(overlap_idx):
            return None
        return dict(
            gt_segments=truncated_gt[overlap_idx],
            gt_labels=video_anno["gt_labels"][overlap_idx],
        )

    def get_class_map(self, class_map_path):
        if not os.path.exists(class_map_path):
            class_map = get_class_index(self.ann_file, class_map_path)
            self.logger(f"Class map is saved in {class_map_path}, total {len(class_map)} classes.")
        else:
            with open(class_map_path, "r", encoding="utf8") as f:
                lines = f.readlines()
            class_map = [item.rstrip("\n") for item in lines]
        return class_map

    def get_gt(self):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        return len(self.data_list)


def empty_annotation():
    """A GT-free annotation, i.e. a background (negative) window."""
    return dict(
        gt_segments=np.zeros((0, 2), dtype=np.float32),
        gt_labels=np.zeros((0,), dtype=np.int32),
    )


def compute_gt_completeness(gt_boxes, anchors):
    """Compute the completeness of the gt_bboxes.
       GT will be first truncated by the anchor start/end, then the completeness is defined as the ratio of the truncated_gt_len / original_gt_len.
       If this ratio is too small, it means this gt is not complete enough to be used for training.
    Args:
        gt_boxes: np.array shape [N, 2]
        anchors:  np.array shape [2]
    """

    scores = np.zeros(gt_boxes.shape[0])  # initialized as 0
    valid_idx = np.logical_and(gt_boxes[:, 0] < anchors[1], gt_boxes[:, 1] > anchors[0])  # valid gt
    valid_gt_boxes = gt_boxes[valid_idx]

    truncated_valid_gt_len = np.minimum(valid_gt_boxes[:, 1], anchors[1]) - np.maximum(valid_gt_boxes[:, 0], anchors[0])
    original_valid_gt_len = np.maximum(valid_gt_boxes[:, 1] - valid_gt_boxes[:, 0], 1e-6)
    scores[valid_idx] = truncated_valid_gt_len / original_valid_gt_len

    # also truncated gt
    truncated_gt_boxes = np.stack(
        [np.maximum(gt_boxes[:, 0], anchors[0]), np.minimum(gt_boxes[:, 1], anchors[1])], axis=1
    )
    return scores, truncated_gt_boxes  # shape [N]


# ── merged from util.py ──
import numpy as np


def filter_same_annotation(annotation):
    gt_segments = []
    gt_labels = []
    gt_both = []
    for gt_segment, gt_label in zip(annotation["gt_segments"].tolist(), annotation["gt_labels"].tolist()):
        if (gt_segment, gt_label) not in gt_both:
            gt_segments.append(gt_segment)
            gt_labels.append(gt_label)
            gt_both.append((gt_segment, gt_label))
        else:
            continue

    annotation = dict(
        gt_segments=np.array(gt_segments, dtype=np.float32),
        gt_labels=np.array(gt_labels, dtype=np.int32),
    )
    return annotation


if __name__ == "__main__":
    anno1 = dict(gt_segments=np.array([[3, 5], [3, 6], [3, 5]]), gt_labels=np.array([0, 1, 0]))
    print(filter_same_annotation(anno1))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 6.]], dtype=float32),
    # 'gt_labels': array([0, 1], dtype=int32)}

    anno2 = dict(gt_segments=np.array([[3, 5], [3, 6], [3, 5]]), gt_labels=np.array([0, 1, 2]))
    print(filter_same_annotation(anno2))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 6.], [3., 5.]], dtype=float32),
    # 'gt_labels': array([0, 1, 2], dtype=int32)}

    anno3 = dict(gt_segments=np.array([[3, 5], [3, 5], [3, 5]]), gt_labels=np.array([0, 1, 1]))
    print(filter_same_annotation(anno3))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 5.]], dtype=float32),
    # 'gt_labels': array([0, 1], dtype=int32)}


# ── merged from sliding.py ──
import json
import numpy as np
import math
import random
from copy import deepcopy
from .builder import DATASETS


@DATASETS.register_module()
@DATASETS.register_module(name="ThumosSlidingDataset")  # deprecated alias (renamed -> PlainSlidingDataset)
class PlainSlidingDataset(SlidingWindowDataset):
    """Plain uniform sliding-window dataset over the (original) videos.

    Windows are placed by stride only (window_overlap_ratio), label-free — so it is
    valid for val/test/inference. The former name ``ThumosSlidingDataset`` stays
    registered as a back-compat alias for existing configs (remote boxes hold configs
    that still spell it), and is also bound as a module-level symbol below so
    ``from vtrace.datasets.sliding import ThumosSlidingDataset`` keeps working.
    The module itself was ``thumos.py`` until the THUMOS-specific components were
    dropped; nothing here is benchmark-specific any more.
    """

    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            # gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            # gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])
            gt_start = anno["frame_segment"][0] if "frame_segment" in anno else int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = anno["frame_segment"][1] if "frame_segment" in anno else int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        entry = self.data_list[index]
        video_name, video_info, window_anno, window = entry[:4]
        if len(entry) > 4:
            window_anno, window = self._resolve_jitter(entry)
        return self._build_sample(video_name, video_info, window_anno, window)

    def _resolve_jitter(self, entry):
        """Re-draw a jittered window's position, for this read only.

        Position diversity is an augmentation, so it is drawn per read rather than
        materialized as extra windows: a fixed grid shows every bout at exactly one
        phase, forever, while the same window count re-drawn each epoch eventually shows
        it at all of them. Uses the stdlib ``random`` module like every other
        augmentation in the pipeline, so it inherits the DataLoader's per-worker seeding
        and needs no epoch hook (which ``persistent_workers`` would not see anyway).

        A draw whose window would be *dropped* (no GT, without keep_empty_gt) falls back
        to the stored window, so jitter never changes which windows an epoch contains —
        only where they sit.
        """
        lo, hi, centers, video_anno, gt_mode = entry[4]
        start = random.randint(lo, hi)
        window = centers[start:start + self.window_size]
        window_anno = self.window_annotation(video_anno, window, gt_mode=gt_mode)
        if window_anno is None:
            return entry[2], entry[3]
        return window_anno, window

    def _build_sample(self, video_name, video_info, video_anno, window_snippet_centers):
        """Turn one resolved window into a pipeline input.

        Split out of ``__getitem__`` so a subclass can decide the window's position at
        load time (per-epoch phase jitter) instead of only at build time.
        """
        if video_anno != {}:
            video_anno = deepcopy(video_anno)  # avoid modify the original dict
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        sample = dict(
            video_name=video_name,
            data_path=self.data_path,
            window_size=self.window_size,
            # trunc window setting
            feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
            feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
            sample_stride=self.sample_stride,
            # sliding post process setting
            fps=video_info["frame"] / video_info["duration"],
            snippet_stride=self.snippet_stride,
            window_start_frame=window_snippet_centers[0],
            duration=video_info["duration"],
            offset_frames=self.offset_frames,
            # training setting
            **video_anno,
        )
        # Virtual-clip metadata (omitted for legacy physical clips). When set,
        # the pipeline reads frames from `source_video` with an offset of
        # `source_frame_offset` instead of opening data_path/<name>.mp4.
        if "source_video" in video_info:
            sample["source_video"] = video_info["source_video"]
            sample["source_frame_offset"] = video_info.get("source_frame_offset", 0)
            sample["clip_frame_count"] = video_info["frame"]
            # PTS-table reference (added in Phase 2 of the refactor in
            # pts-based-frame-mapping.md (archived)). Consumed by
            # `convert_to_seconds` to map model-frame indices back to
            # clip-relative seconds without assuming CFR. Missing on
            # virtual-clip datasets prepped before the PTS upgrade — those
            # fall back to the `fps`-based path.
            if "source_pts_table" in video_info:
                sample["source_pts_table"] = video_info["source_pts_table"]
        # Decode proxy: a downscaled, frame-aligned copy of `source_video`.
        # Only the file frames come from changes — `source_frame_offset` still
        # applies, so no `decode_frame_offset` override here.
        if "proxy_video" in video_info:
            sample["proxy_video"] = video_info["proxy_video"]
        # Legacy per-window clip cache. No longer written, still read so
        # datasets prepared before the proxy refactor keep loading.
        if "cached_video" in video_info:
            sample["cached_video"] = video_info["cached_video"]
            sample["decode_frame_offset"] = 0
        results = self.pipeline(sample)
        return results


@DATASETS.register_module()
class BehaviorTargetedSlidingDataset(PlainSlidingDataset):
    """Behavior-aware *targeted* temporal sampling (train side only).

    Designed for sliding over the ORIGINAL videos (which may be hours long), not over
    a cache of clips pre-cut to window_size — a clip exactly as long as the window has
    no slack for the dense pass to slide into, so targeted sampling degenerates to the
    plain sampler there.

    Instead of uniform overlap (which densifies ALL behaviors — mostly the frequent
    ones — plus background), this builds the train window set as:
      (1) a SPARSE base pass over the whole video at the config's window_overlap_ratio
          stride (default overlap0 => stride=window_size) so every region is covered
          once (keeps train dist ~ the full-video eval dist), PLUS
      (2) `rare_phases` ANCHOR windows per rare-class GT segment, each of which draws a
          fresh random position inside its own phase stratum on every ``__getitem__``.
          The strata partition the range from "the bout has just entered at the window
          tail" to "it is just leaving at the head", so k anchors cover k different
          phases and, across epochs, a continuum of positions within each.

    Phase diversity is an augmentation, so it is applied at load time rather than
    materialized: the earlier version enumerated a whole sweep at an `rare_overlap`
    stride, costing ~W/stride windows per bout — several times the base pass — to buy a
    fixed grid of phases repeated identically every epoch.
    `rare_phases` anchors cost k windows per bout — a count independent of window_size,
    stride and bout length — and give a *different* phase every epoch. The jitter uses
    the stdlib ``random`` module like every other augmentation in the pipeline, so it
    inherits the DataLoader's per-worker seeding and needs no epoch hook.

    Anchor windows always carry the TRUNCATED GT, whatever `window_gt_mode` says (a
    partial bout is correctly labeled for a per-frame head, and the completeness rule
    would reject exactly the entering/leaving reads the strata exist for). Base-pass
    windows follow the inherited settings: `window_gt_mode` (default "truncate", i.e.
    the same semantics as the anchors), `keep_empty_gt` for windows overlapping no GT,
    and `base_jitter` if the config wants the grid to move as well. Anchor positions are
    not de-duplicated against the base pass — they move every epoch, so there is nothing
    stable to de-duplicate against.

    Both parts need slack: a bout can only shift phase if the video extends ~one window
    before and after it. Entries cut to exactly `window_size` (a 768-frame clip cache)
    have none, and this degenerates to the plain sampler.

    WHICH classes are rare is **auto-detected** from this subset's annotations (see
    `_detect_rare_classes`) — it is a property of the dataset, not a tuning knob, so
    there is deliberately no config option for it. Passing `rare_classes=` raises.

    Eval/val datasets MUST stay plain PlainSlidingDataset (full-video sliding) — this
    only reshapes the TRAIN sampling.
    """

    # A class counts as rare when it is scarce RELATIVE TO THE CORPUS'S OWN AVERAGE
    # class (a tail exists only by comparison), OR scarce relative to the single most
    # frequent class.
    #
    # Deliberately not an absolute prevalence cutoff. A fixed threshold is blind to the
    # class count: with a labeled fraction F spread over C classes the mean class
    # prevalence is F/C, so on a sparsely labeled corpus escaping the cutoff demands an
    # impossible share of all labeled frames and every class gets flagged — including
    # the most frequent one, which is then oversampled as if it were the tail. The
    # criterion below is scale-free in both directions: it has no opinion about how
    # densely a corpus is labeled, only about how uneven it is.
    RARE_SCARCITY_FRAC = 0.5
    RARE_HEAD_SHARE_MAX = 0.25
    # Set on `LegacyBehaviorTargetedSlidingDataset` only, to reproduce runs that
    # predate the switch. None means "use RARE_SCARCITY_FRAC instead".
    RARE_ABSOLUTE_PREVALENCE = None

    # Per-class anchor quota, ON BY DEFAULT. `rare_phases` alone gives every rare bout
    # the same number of anchors, so a class's oversampling comes out proportional to
    # its BOUT COUNT — the wrong axis, and one that can widen the very deficit the
    # anchors exist to close when a rare class also has few bouts. Scaling by frames
    # instead makes exposure track "how much of this behavior has the model seen".
    DEFAULT_RARE_QUOTA = dict(mode="inv_freq_pow", power=0.5, max_phases=12)

    def __init__(self, *args, rare_phases=2, rare_jitter=True, rare_overlap=None,
                 rare_quota="default", **kwargs):
        if "rare_classes" in kwargs:
            raise TypeError(
                "BehaviorTargetedSlidingDataset auto-detects the rare classes from the "
                "annotations; `rare_classes` is not configurable. Remove it from the "
                "dataset config (tune `rare_phases` if you want more/fewer rare windows)."
            )
        # Windows per rare bout. This IS the rare-class oversampling factor, and it is
        # now the only thing that sets it — no longer a side effect of window_size/stride.
        self.rare_phases = max(1, int(rare_phases))
        # Off => every anchor stays pinned at its stratum centre. For debugging and for
        # reproducing a run window-for-window; training wants it on.
        self.rare_jitter = bool(rare_jitter)
        # Retired: the window count no longer derives from a stride. Accepted rather than
        # rejected because remote boxes hold configs that still pass it; warned about in
        # the sampling report (self.logger does not exist yet at this point).
        self._retired_rare_overlap = rare_overlap
        # Per-class anchor quota. `rare_quota=dict(mode="inv_freq_pow", power=0.5,
        # max_phases=12)` scales each rare class's phase count by (ref/frames_c)**power,
        # `ref` being the LARGEST rare class, so no class ever gets fewer anchors than
        # `rare_phases` and the change is one-directional. power=0.5 mirrors the
        # inv_freq_sqrt loss weighting; full equalisation (power=1) would redraw hundreds
        # of windows from a handful of events, which is memorisation rather than coverage.
        # "default" -> DEFAULT_RARE_QUOTA; None -> explicitly off (uniform phases).
        self.rare_quota = (self.DEFAULT_RARE_QUOTA if rare_quota == "default"
                           else rare_quota)
        self._rare_phases_by_class = {}
        self.rare_classes = set()  # filled by get_dataset() before any window is built
        super().__init__(*args, **kwargs)
        self._log_sampling_report()

    def get_dataset(self):
        # Detection must precede window construction: split_video_to_windows() reads
        # self.rare_classes for every video.
        self._source_stats = self._collect_source_stats()
        self.rare_classes = self._detect_rare_classes()
        self._rare_phases_by_class = self._resolve_rare_quota()
        self._anchor_windows = 0
        # What the sparse base pass alone would have produced, accumulated as the
        # real window set is built. This is the "before" side of the report: the
        # phase sweep is *added to* these windows, so the delta is the enhancement.
        self._base_counts = self._zero_counts()
        super().get_dataset()

    def _num_frames(self, video_info):
        return int(video_info["duration"] * self.fps) if self.fps > 0 else video_info["frame"]

    def _blocked_videos(self):
        if isinstance(self.block_list, list):
            return set(self.block_list)
        if self.block_list is not None:
            with open(self.block_list, "r") as f:
                return set(line.rstrip("\n") for line in f)
        return set()

    def _collect_source_stats(self):
        """One pass over the annotation file for everything the report needs.

        Returns the *source* picture — what is in the dataset before any window
        sampling — so the report can separate "what the corpus contains" from
        "what the sampler produced".
        """
        with open(self.ann_file, "r") as f:
            anno_database = json.load(f)["database"]
        blocked = self._blocked_videos()
        stride = max(1, self.snippet_stride)

        stats = {
            "entries": 0,
            "videos": set(),
            "total_snippets": 0.0,
            "duration_sec": 0.0,
            "labeled_snippets": {c: 0.0 for c in range(len(self.class_map))},
            "segments": {c: 0 for c in range(len(self.class_map))},
            "decode": {"proxy": 0, "cached_clip": 0, "source": 0, "clip_file": 0},
        }
        for video_name, video_info in anno_database.items():
            if (video_name in blocked) or (video_info["subset"] not in self.subset_name):
                continue
            stats["entries"] += 1
            stats["videos"].add(video_info.get("source_video") or video_name)
            stats["total_snippets"] += len(np.arange(0, self._num_frames(video_info), stride))
            stats["duration_sec"] += float(video_info.get("duration", 0.0))
            if video_info.get("proxy_video"):
                stats["decode"]["proxy"] += 1
            elif video_info.get("cached_video"):
                stats["decode"]["cached_clip"] += 1
            elif video_info.get("source_video"):
                stats["decode"]["source"] += 1
            else:
                stats["decode"]["clip_file"] += 1

            if self.test_mode:
                continue
            video_anno = self.get_gt(video_info)
            if video_anno is None:
                continue
            for (a, b), lab in zip(video_anno["gt_segments"], video_anno["gt_labels"]):
                lab = int(lab)
                if lab in stats["labeled_snippets"]:
                    stats["labeled_snippets"][lab] += max(0.0, (float(b) - float(a)) / stride)
                    stats["segments"][lab] += 1
        return stats

    def _zero_counts(self):
        """Per-class tallies of what the sampler handed to the model."""
        return {
            "windows": {c: 0 for c in range(len(self.class_map))},
            "instances": {c: 0 for c in range(len(self.class_map))},
            "snippets": {c: 0.0 for c in range(len(self.class_map))},
            "total_windows": 0,
            "empty_windows": 0,
        }

    def _accumulate(self, counts, window_anno):
        counts["total_windows"] += 1
        labels = window_anno.get("gt_labels", []) if window_anno else []
        if len(labels) == 0:
            counts["empty_windows"] += 1
            return
        stride = max(1, self.snippet_stride)
        seen = set()
        for (a, b), lab in zip(window_anno["gt_segments"], labels):
            lab = int(lab)
            if lab not in counts["instances"]:
                continue
            counts["instances"][lab] += 1
            counts["snippets"][lab] += max(0.0, (float(b) - float(a)) / stride)
            seen.add(lab)
        for lab in seen:
            counts["windows"][lab] += 1

    def _detect_rare_classes(self):
        """Pick the rare classes from this subset's GT — no config knob.

        Two criteria (either one is enough), both scale-free so they transfer
        across corpora with wildly different label density:
          * prevalence  = labeled snippets of class c / all snippets in the subset
                          — only consulted on a sparsely-labeled corpus, see
                          DENSE_CORPUS_LABELED_FRAC; a fixed prevalence cutoff is
                          blind to the class count and fires on most classes once
                          the timeline is densely labeled.
          * head share  = labeled snippets of class c / labeled snippets of the
                          most frequent class
        Classes with no GT in this subset are skipped (nothing to densify).
        """
        if self.test_mode:
            self.logger("[BehaviorTargeted] test_mode: no GT to inspect, rare_classes={} (plain sparse sampling)")
            return set()

        pos = self._source_stats["labeled_snippets"]
        segs = self._source_stats["segments"]
        total = self._source_stats["total_snippets"]

        head = max(pos.values()) if pos else 0.0
        if total <= 0 or head <= 0:
            self.logger("[BehaviorTargeted] no GT found in subset '%s': rare_classes={} (plain sparse sampling)" % self.subset_name)
            return set()

        labeled_frac = sum(pos.values()) / total
        present = [c for c, f in pos.items() if f > 0]
        # The corpus's own average class, not a constant: on a 5-class corpus that
        # labels 17% of frames, an "average" class holds 3.4% of the timeline, and
        # scarce has to mean scarce against THAT.
        mean_prev = (labeled_frac / len(present)) if present else 0.0
        legacy_abs = self.RARE_ABSOLUTE_PREVALENCE
        thr = (legacy_abs if legacy_abs is not None
               else self.RARE_SCARCITY_FRAC * mean_prev)

        rare = set()
        for c, frames in pos.items():
            if frames <= 0:
                continue
            if (frames / total < thr) or (frames / head <= self.RARE_HEAD_SHARE_MAX):
                rare.add(c)

        names = self._class_names()
        self.logger(
            "[BehaviorTargeted] auto-detected rare_classes=%s (labeled_frac=%.2f over %d classes "
            "=> mean class prevalence %.4f; thresholds: prevalence<%.4f (%s) or head_share<=%.2f) | %s"
            % (
                sorted(names[c] for c in rare) or "{}",
                labeled_frac, len(present), mean_prev, thr,
                "LEGACY absolute" if legacy_abs is not None
                else "%.2gx the mean class" % self.RARE_SCARCITY_FRAC,
                self.RARE_HEAD_SHARE_MAX,
                ", ".join(
                    "%s: prev=%.4f head_share=%.3f segs=%d%s"
                    % (names[c], pos[c] / total, pos[c] / head, segs[c], " RARE" if c in rare else "")
                    for c in sorted(pos)
                ),
            )
        )
        # Degeneracy guard. Density is irrelevant here: if every class is rare then
        # nothing is being preferred over anything, and the anchors are pure cost.
        if present and len(rare) >= len(present):
            self.logger(
                "[BehaviorTargeted] WARNING: ALL %d labeled classes were flagged rare, so nothing is "
                "being oversampled RELATIVE to anything — every bout gets anchors and the only net "
                "effects are a larger epoch and a background share below the evaluation prior. This "
                "corpus has no tail to correct (labeled_frac=%.2f, mean class prevalence %.4f). Use "
                "PlainSlidingDataset, or raise RARE_SCARCITY_FRAC's selectivity."
                % (len(present), labeled_frac, mean_prev)
            )
        return rare

    def _class_names(self):
        return {i: (self.class_map[i] if i < len(self.class_map) else str(i)) for i in range(len(self.class_map))}

    def split_video_to_windows(self, video_name, video_info, video_anno):
        num_frames = self._num_frames(video_info)
        video_snippet_centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_num = len(video_snippet_centers)
        W = self.window_size
        max_start = max(0, snippet_num - W)
        data_list = []

        # (1) sparse base coverage over the whole video (preserves background, and keeps
        # the train distribution close to the full-video eval distribution).
        base_stride = max(1, self.window_stride)
        base_starts = set(range(0, max_start + 1, base_stride))
        base_starts.add(max_start)  # always cover the tail
        for window_start in sorted(base_starts):
            window = video_snippet_centers[window_start:window_start + W]
            window_anno = self.window_annotation(video_anno, window)
            if window_anno is None:
                continue
            entry = [video_name, video_info, window_anno, window]
            jitter = self._jitter_record(window_start, max_start, video_snippet_centers, video_anno)
            if jitter is not None:
                entry.append(jitter)
            data_list.append(entry)
            self._accumulate(self._base_counts, window_anno)

        # (2) `rare_phases` anchors per rare bout. Each carries a jitter record --
        # (stratum_lo, stratum_hi, all snippet centres of this video, the video's full
        # GT) -- and __getitem__ redraws its position inside the stratum on every read.
        # The stored window is the stratum centre, so the entry is a valid window even
        # with jitter off, and the sampling report can describe it.
        if (video_anno != {}) and self.rare_classes and (len(video_anno["gt_segments"]) > 0):
            for (a, b), lab in zip(video_anno["gt_segments"], video_anno["gt_labels"]):
                if int(lab) not in self.rare_classes:
                    continue
                a_s = int(a // self.snippet_stride)
                b_s = int(b // self.snippet_stride)
                k = self._rare_phases_by_class.get(int(lab), self.rare_phases)
                for slot_lo, slot_hi in self._phase_slots(a_s, b_s, W, max_start, k):
                    anchor = (slot_lo + slot_hi) // 2
                    window = video_snippet_centers[anchor:anchor + W]
                    window_anno = self.truncate_window_gt(video_anno, window[0], window[-1])
                    if window_anno is None:  # the bout cannot fall outside its own range
                        continue
                    data_list.append([
                        video_name, video_info, window_anno, window,
                        # Anchors always truncate, whatever window_gt_mode says: a bout
                        # entering at the tail is the point of the stratum, and the
                        # completeness rule would reject exactly those reads.
                        (slot_lo, slot_hi, video_snippet_centers, video_anno, "truncate"),
                    ])
                    self._anchor_windows += 1
        return data_list

    def _resolve_rare_quota(self):
        """Anchor phases per rare class -- uniform `rare_phases` unless rare_quota is set.

        Scales by labeled-snippet count, not bout count, because exposure is
        phases x bouts x mean_bout_length ~= phases x frames: a class with few long
        bouts and a class with many short ones need different phase counts to reach
        the same exposure.
        """
        q = self.rare_quota
        base = self.rare_phases
        if not q or not self.rare_classes:
            return {c: base for c in self.rare_classes}
        mode = q.get("mode", "inv_freq_pow")
        if mode != "inv_freq_pow":
            raise ValueError(f"unknown rare_quota mode {mode!r} (only 'inv_freq_pow')")
        power = float(q.get("power", 0.5))
        cap = int(q.get("max_phases", 12))
        pos = self._source_stats["labeled_snippets"]
        ref = max((pos.get(c, 0) for c in self.rare_classes), default=0)
        out = {}
        for c in self.rare_classes:
            n = pos.get(c, 0)
            if n <= 0 or ref <= 0:
                out[c] = base
                continue
            out[c] = int(min(cap, max(base, round(base * (ref / n) ** power))))
        return out

    def _phase_slots(self, a_s, b_s, W, max_start, phases=None):
        """Partition a bout's admissible window starts into `rare_phases` strata.

        The full range runs from "the bout has just entered at the window tail" to "it is
        just leaving at the head" (``+2`` / ``-1`` so both ends still overlap the bout by
        >=1 snippet, the completeness test being strict). Confining each anchor's jitter
        to its own stratum makes k anchors cover k *distinct* phases; k iid draws over
        the whole range would cluster instead.
        """
        lo = max(0, min(a_s - W + 2, max_start))
        hi = max(lo, min(b_s - 1, max_start))
        span = hi - lo + 1
        k = int(phases if phases else self.rare_phases)
        slots = []
        for i in range(k):
            s_lo = lo + (span * i) // k
            s_hi = max(s_lo, lo + (span * (i + 1)) // k - 1)
            if (s_lo, s_hi) not in slots:  # span < k => strata collapse
                slots.append((s_lo, s_hi))
        return slots

    def _resolve_jitter(self, entry):
        # rare_jitter=False pins every anchor to its stratum centre (the stored window),
        # which is what reproducing a run window-for-window needs. It does not disable
        # base_jitter — that is the base pass' own setting.
        if not self.rare_jitter and entry[4][4] == "truncate":
            return entry[2], entry[3]
        return super()._resolve_jitter(entry)

    @staticmethod
    def _ratio(after, before):
        """'x2.6' / 'new' / '--' — the multiplier the anchors bought for a category."""
        if before > 0:
            return "x%.1f" % (after / before)
        return "new" if after > 0 else "--"

    def _final_counts(self):
        counts = self._zero_counts()
        for entry in self.data_list:
            anno = entry[2]
            self._accumulate(counts, anno if anno != {} else {})
        return counts

    def _log_sampling_report(self):
        """Log what the corpus holds and what targeted sampling did to it.

        Two blocks: the dataset as prepared (source of truth, unaffected by
        sampling), then per-category before/after for the sampler. "Before" is
        the sparse base pass alone — i.e. plain sliding at this
        window_overlap_ratio — which is exactly what the anchors are added to.
        Anchor rows describe the stratum centres; the phase actually trained on is
        redrawn inside each stratum on every read, so the per-class *counts* are
        exact but the segment positions shift epoch to epoch.
        """
        log = self.logger
        names = self._class_names()
        stats = getattr(self, "_source_stats", None) or {}
        before, after = self._base_counts, self._final_counts()

        total_snips = stats.get("total_snippets", 0.0)
        labeled = sum(stats.get("labeled_snippets", {}).values())
        segments = sum(stats.get("segments", {}).values())
        decode = stats.get("decode", {})
        decode_desc = ", ".join(f"{k}={v}" for k, v in decode.items() if v) or "n/a"

        log("[BehaviorTargeted] ===== dataset =====")
        log("[BehaviorTargeted]   subset=%s  entries=%d  source videos=%d  classes=%d [%s]"
            % (self.subset_name, stats.get("entries", 0), len(stats.get("videos", ())),
               len(self.class_map), ", ".join(self.class_map)))
        log("[BehaviorTargeted]   timeline=%d snippets (%.1f min)  labeled=%d (%.2f%%)  GT segments=%d"
            % (total_snips, stats.get("duration_sec", 0.0) / 60.0, labeled,
               100.0 * labeled / total_snips if total_snips else 0.0, segments))
        log("[BehaviorTargeted]   window=%d  base_stride=%d  snippet_stride=%d  fps=%s  keep_empty_gt=%s"
            % (self.window_size, self.window_stride, self.snippet_stride,
               self.fps, self.keep_empty_gt))
        log("[BehaviorTargeted]   base pass: window_gt_mode=%s%s  base_jitter=%s"
            % (self.window_gt_mode,
               "" if self.window_gt_mode == "truncate" else " (ioa_thresh=%.2f)" % self.ioa_thresh,
               ("+/-%d snippets (%.0f%% of window)"
                % (max(1, int(self.base_jitter * self.window_size)), 100 * self.base_jitter))
               if self.base_jitter > 0 else "off"))
        log("[BehaviorTargeted]   decode source: %s" % decode_desc)

        log("[BehaviorTargeted] ===== targeted sampling: base pass -> + rare-class anchors =====")
        log("[BehaviorTargeted]   rare_phases=%d  jitter=%s  anchor windows=%d (%d per rare bout)"
            % (self.rare_phases, "on" if self.rare_jitter else "OFF (pinned to stratum centres)",
               self._anchor_windows, self.rare_phases))
        if self._retired_rare_overlap is not None:
            log("[BehaviorTargeted]   NOTE: `rare_overlap=%s` is retired and ignored. Window count "
                "is now rare_phases x rare bouts, independent of window_size and stride; phase "
                "diversity comes from per-read jitter instead of a materialized sweep. Set "
                "`rare_phases` (currently %d) to change the rare-class oversampling factor."
                % (self._retired_rare_overlap, self.rare_phases))
        def cell(b, a, width=24):
            return ("%d -> %d %s" % (b, a, self._ratio(a, b))).ljust(width)

        log("[BehaviorTargeted]   %-16s %-5s %-24s %-24s %s"
            % ("category", "rare", "windows", "instances", "labeled snippets"))
        for c in range(len(self.class_map)):
            if not (after["windows"][c] or before["windows"][c] or stats.get("segments", {}).get(c)):
                continue  # class absent from this subset — nothing to densify
            log("[BehaviorTargeted]   %-16s %-5s %s %s %s"
                % (names[c], "YES" if c in self.rare_classes else "-",
                   cell(before["windows"][c], after["windows"][c]),
                   cell(before["instances"][c], after["instances"][c]),
                   cell(int(before["snippets"][c]), int(after["snippets"][c]))))
        log("[BehaviorTargeted]   %-16s %-5s %s (empty/background %d -> %d)"
            % ("TOTAL", "", cell(before["total_windows"], after["total_windows"]),
               before["empty_windows"], after["empty_windows"]))
        if self.rare_quota and self.rare_classes:
            log("[BehaviorTargeted]   rare_quota=%s -> per-class phases: %s"
                % (self.rare_quota,
                   ", ".join("%s=%d" % (names[c], self._rare_phases_by_class[c])
                             for c in sorted(self.rare_classes))))
        if not self.rare_classes:
            log("[BehaviorTargeted]   no rare classes -> no anchors, this is plain sparse sampling")




@DATASETS.register_module()
class LegacyBehaviorTargetedSlidingDataset(BehaviorTargetedSlidingDataset):
    """`BehaviorTargetedSlidingDataset` with the pre-scarcity-criterion detector.

    Exists only so that runs predating the switch stay reproducible: the absolute
    `prevalence < 0.10` cutoff, applied unconditionally, plus uniform anchor phases
    (no quota). Point a config at this class to re-score or extend a run produced
    under the old settings.

    Why this matters concretely: the criterion changes how many windows an epoch has,
    not just which ones — flagging more classes multiplies the anchor count. A config
    written against the old behaviour and run against the new one can therefore lose a
    large fraction of its gradient updates per epoch, silently — which is not a
    sampling-distribution difference but a training-budget one. Any comparison against
    such a run has to pin the detector, hence this class.
    """

    RARE_ABSOLUTE_PREVALENCE = 0.10   # unconditional, as it was

    def __init__(self, *args, rare_quota=None, **kwargs):
        # Uniform phases: the quota is a post-hoc fix and no legacy run had it.
        super().__init__(*args, rare_quota=rare_quota, **kwargs)


# Deprecated module-level alias, matching the registry alias on PlainSlidingDataset.
ThumosSlidingDataset = PlainSlidingDataset
