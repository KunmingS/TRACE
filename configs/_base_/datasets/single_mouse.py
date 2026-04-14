annotation_path = "/media/SSD_4T/skm/test_videos/figure5/WIN_20251109_20_02_06_left_cut/dataset.json"
class_map = "/media/12T_3ab/skm/TAD_models/OpenTAD/data/CALMS21/annotations/single_4.txt"
data_path = "/media/SSD_4T/skm/test_videos/figure5/WIN_20251109_20_02_06_left_cut/clips"
block_list = None

window_size = 256

dataset = dict(
    train=dict(
        type="ThumosPaddingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=1,
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="VideoInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.5,
                crop_ratio=[0.9, 1.0],
            ),
            dict(type="VideoDecode"),
            dict(type="VideoResize", scale=(-1, 256)),
            dict(type="VideoRandomResizedCrop"),
            dict(type="VideoResize", scale=(224, 224)),
            dict(type="VideoFlip", flip_ratio=0.5),
            dict(type="VideoFormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="VideoInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="VideoDecode"),
            dict(type="VideoResize", scale=(-1, 224)),
            dict(type="VideoCenterCrop", crop_size=224),
            dict(type="VideoFormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="VideoInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="VideoDecode"),
            dict(type="VideoResize", scale=(-1, 224)),
            dict(type="VideoCenterCrop", crop_size=224),
            dict(type="VideoFormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="Precision",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
    gt_fps=30.0,
    eval_fps=30.0,
)
