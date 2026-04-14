annotation_path = "/media/12T_3ab/skm/test_videos/calms21/task1_no_other.json"
class_map = "data/CALMS21/category_idx.txt"
data_path = "/media/12T_3ab/skm/test_videos/calms21/task1_videos_clip_mp4"
block_list = None

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
        window_overlap_ratio=0.25,
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
        window_overlap_ratio=0.5,
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
