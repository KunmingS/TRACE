import json
import torch
from torch.utils.data.dataloader import default_collate
from collections.abc import Sequence

from ..registry import Registry

DATASETS = Registry("dataset")
PIPELINES = Registry("pipelines")


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict."""
    cfg = dict(cfg)
    if default_args:
        for k, v in default_args.items():
            cfg.setdefault(k, v)
    return DATASETS.build(cfg)


def build_dataloader(dataset, batch_size, shuffle=False, drop_last=False, **kwargs):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate,
        pin_memory=True,
        **kwargs,
    )
    return dataloader


def collate(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    gpu_stack_keys = ["inputs", "masks"]

    collate_data = {}
    for key in batch[0]:
        if key in gpu_stack_keys:
            collate_data[key] = default_collate([sample[key] for sample in batch])
        else:
            collate_data[key] = [sample[key] for sample in batch]
    return collate_data


def get_class_index(gt_json_path, class_map_path):
    with open(gt_json_path, "r") as f:
        anno = json.load(f)

    anno = anno["database"]
    class_map = []
    for video_name in anno.keys():
        if "annotations" in anno[video_name]:
            for tmpp_data in anno[video_name]["annotations"]:
                if tmpp_data["label"] not in class_map:
                    class_map.append(tmpp_data["label"])

    class_map.sort()
    with open(class_map_path, "w") as f:
        for name in class_map:
            f.write(name + "\n")
    return class_map
