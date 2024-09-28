#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from .GAGAvatar import GAGAvatar

def build_model(model_cfg, ):
    model_dict = {
        'GAGAvatar': GAGAvatar,
    }
    return model_dict[model_cfg.NAME](model_cfg, )
