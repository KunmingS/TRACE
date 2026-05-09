import json
import os
class Compose:
    # Defined as a top-level class (not a closure) so dataset instances remain
    # picklable for DataLoader workers under spawn/forkserver start methods
    # (Python 3.14 default on Linux, always on macOS/Windows).
    def __init__(self, pipeline):
        from ..builder import PIPELINES
        self.transforms = [PIPELINES.build(t) if isinstance(t, dict) else t for t in pipeline]

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
        return results

from ..builder import DATASETS, get_class_index


@DATASETS.register_module()
class PaddingDataset:
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
        # for data oversampling
        oversample_ratios=None,  # dict of class_name: ratio for oversampling minority classes
        # for feature setting
        feature_stride=-1,  # the frames between two adjacent features, such as 4 frames
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        offset_frames=0,  # the start offset frame of the input feature
        fps=-1,  # some annotations are based on video-seconds
        logger=None,
    ):
        super(PaddingDataset, self).__init__()

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
        self.pipeline = Compose(pipeline)

        # oversampling settings
        self.oversample_ratios = oversample_ratios

        # feature settings
        self.feature_stride = feature_stride
        self.sample_stride = sample_stride
        self.offset_frames = int(offset_frames)
        self.snippet_stride = int(feature_stride * sample_stride)
        self.fps = fps

        self.get_dataset()

        # Apply oversampling if specified
        if self.oversample_ratios is not None and not self.test_mode:
            self.apply_oversampling()

        self.logger(f"{self.subset_name} subset: {len(self.data_list)} videos")

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
                    continue

            self.data_list.append([video_name, video_info, video_anno])
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def apply_oversampling(self):
        """Oversample minority classes by duplicating samples.

        oversample_ratios: dict of class_name -> ratio (e.g., {'attack': 6.0, 'mount': 6.0})
        A ratio of 6.0 means the class will appear 6x more in the training data.
        """
        if self.oversample_ratios is None:
            return

        original_length = len(self.data_list)

        # Count samples per class and identify which samples belong to which class
        class_samples = {class_name: [] for class_name in self.class_map}

        for idx, (video_name, video_info, video_anno) in enumerate(self.data_list):
            # Get the dominant class in this video (the one with most frames)
            if 'gt_labels' in video_anno and len(video_anno['gt_labels']) > 0:
                labels = video_anno['gt_labels']
                # Count each class
                from collections import Counter
                label_counts = Counter(labels)
                # Get the most common class
                dominant_class_idx = label_counts.most_common(1)[0][0]
                dominant_class_name = self.class_map[dominant_class_idx]
                class_samples[dominant_class_name].append(idx)

        # Calculate how many times to duplicate each sample
        new_data_list = list(self.data_list)  # Start with original data

        for class_name, ratio in self.oversample_ratios.items():
            if class_name not in class_samples:
                self.logger(f"Warning: class '{class_name}' not found in class_map")
                continue

            sample_indices = class_samples[class_name]
            if len(sample_indices) == 0:
                continue

            # Duplicate samples (ratio-1) times (since we already have 1x)
            num_duplicates = int(ratio) - 1
            for _ in range(num_duplicates):
                for idx in sample_indices:
                    new_data_list.append(self.data_list[idx])

        self.data_list = new_data_list
        self.logger(f"Oversampling applied: {original_length} -> {len(self.data_list)} videos")
        self.logger(f"Oversample ratios: {self.oversample_ratios}")

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
