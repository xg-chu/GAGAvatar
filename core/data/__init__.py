#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from .loader_track import TrackedData, DriverData

def build_dataset(data_cfg, split, cross_id=False):
    dataset_dict = {
        'TrackedData': TrackedData,
    }
    return dataset_dict[data_cfg.LOADER](data_cfg, split, cross_id)
