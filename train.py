#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import argparse
import lightning
import numpy as np
import torchvision
from tqdm import tqdm

from core.data import build_dataset
from core.models import build_model
from core.libs.utils import (
    ConfigDict, rtqdm, device_parser, 
    calc_parameters, biuld_logger, calc_psnr, calc_ssim
)

def train(config, dataset, base_model, devices, debug=False):
    # build config
    meta_cfg = ConfigDict(
        model_config_path=os.path.join('./configs/model', f'{config}.yaml'), 
        data_config_path=os.path.join('./configs/data', f'{dataset}.yaml')
    )
    lightning.fabric.seed_everything(42)
    target_devices = device_parser(devices)
    assert len(target_devices) ==  1, f'Only support single GPU training: {target_devices}'
    print(str(meta_cfg))
    # setup model and optimizer
    model = build_model(model_cfg=meta_cfg.MODEL)
    optimizer, scheduler = model.configure_optimizers(meta_cfg.OPTIMIZE)
    op_para_num, all_para_num = calc_parameters(model)
    print('Number of parameters: {:.2f}M / {:.2f}M.'.format(op_para_num/1000000, all_para_num/1000000))
    if base_model is not None:
        assert os.path.exists(base_model), f'Base model not found: {base_model}.'
        model.load_state_dict(torch.load(base_model, map_location='cpu', weights_only=True)['model'], strict=False)
        print('Load base model from: {}.'.format(base_model))
    # load dataset
    train_dataset = build_dataset(data_cfg=meta_cfg.DATASET, split='train')
    val_dataset = build_dataset(data_cfg=meta_cfg.DATASET, split='val')
    val_dataset.slice(16)
    print(f'Train Dataset: {len(train_dataset)}, Val Dataset: {len(val_dataset)}.')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=meta_cfg.TRAIN.BATCH_SIZE, num_workers=meta_cfg.TRAIN.BATCH_SIZE, shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=1, shuffle=False,
    )
    lightning_trainer = Trainer(
        meta_cfg, model, optimizer, scheduler,
        train_dataloader, val_dataloader,
        devices=target_devices, debug=debug,
    )
    lightning_trainer.run_fit()


class Trainer:
    def __init__(
            self, meta_cfg, model, optimizer, scheduler,
            train_dataloader, val_dataloader, devices, debug=False, 
        ):
        self._debug = debug
        self._meta_cfg, self._best_metric = meta_cfg, None
        self._dump_dir = 'outputs' if debug else \
                         os.path.join('outputs', meta_cfg.TRAIN.EXP_STR, meta_cfg.TRAIN.TIME_STR,)
        if not debug:
            os.makedirs(os.path.join(self._dump_dir, 'examples'), exist_ok=False)
            os.makedirs(os.path.join(self._dump_dir, 'checkpoints'), exist_ok=True)
            self.logger = biuld_logger(os.path.join(self._dump_dir, 'train_log.txt'), name=f'train_{meta_cfg.TRAIN.TIME_STR}')
            self.logger.debug(meta_cfg._raw_string)
        else:
            self.logger = biuld_logger(os.path.join(self._dump_dir, 'debug.txt'), name=f'train_{meta_cfg.TRAIN.TIME_STR}')
        # build trainer
        self.lightning_fabric = lightning.Fabric(
            accelerator='cuda', strategy='auto', devices=devices, #precision='16-mixed' 
        )
        self.lightning_fabric.launch()
        # loop config
        self._log_interval = 100
        self._total_iters = meta_cfg.TRAIN.TRAIN_ITER
        self._check_interval = meta_cfg.TRAIN.CHECK_INTERVAL if not debug else 50
        
        # training materials
        self.scheduler = scheduler
        self.model, self.optimizer = self.lightning_fabric.setup(model, optimizer)
        self.train_dataloader = self.lightning_fabric.setup_dataloaders(train_dataloader)
        self.val_dataloader = self.lightning_fabric.setup_dataloaders(val_dataloader)

    def run_fit(self, ):
        # build bar
        fit_bar = tqdm(range(1, self._total_iters+1)) if self._debug else \
                  rtqdm(range(1, self._total_iters+1))
        train_iter = iter(self.train_dataloader)
        self.model.train()
        for iter_idx in fit_bar:
            # get data and prepare
            try:
                batch_data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch_data = next(train_iter)
            # forward
            train_frac = np.clip((iter_idx - 1) / (self._total_iters - 1), 0, 1)
            render_results = self.model(batch_data, train_frac=train_frac, rand=True)
            loss_metrics, show_metric = self.model.calc_metrics(render_results)
            loss = sum(loss_metrics.values())
            self.lightning_fabric.backward(loss)
            # for param in self.model.parameters():
            #     param.grad.nan_to_num_()
            # backward and step
            with torch.no_grad():
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                # logger
                self._logger(iter_idx, fit_bar, loss_metrics, show_metric)
                # checkpoints
                if iter_idx % self._check_interval == 0 or iter_idx == self._total_iters:
                    self.run_val(iter_idx)
                    self._save_checkpoints('latest.pt')

    @torch.no_grad()
    def run_val(self, iter_idx, save_ckpt=True):
        val_iter = iter(self.val_dataloader)
        _validation_outputs = []
        self.model.eval()
        for idx, batch_data in enumerate(val_iter):
            render_results = self.model(batch_data, rand=False)
            gt_rgb = render_results['t_image'].clamp(0, 1).cpu()
            pred_rgb = render_results['sr_gen_image'].clamp(0, 1).cpu() 
            psnr = float(calc_psnr(pred_rgb, gt_rgb, data_range=(0.0, 1.0)))
            ssim = float(calc_ssim(pred_rgb, gt_rgb, data_range=(0.0, 1.0)))
            # visulize
            gt_rgb[:, :, -150:, -150:] = self._resize(batch_data['f_image'].clamp(0, 1).cpu(), (150, 150))
            pred_gs_rgb = render_results['gen_image'].clamp(0, 1).cpu()
            visulize_rgbs = torchvision.utils.make_grid(torch.cat([gt_rgb, pred_gs_rgb, pred_rgb]), nrow=3, padding=0)  
            visulize_rgbs = self._resize(visulize_rgbs, 256)
            _validation_outputs.append({'PSNR': psnr, 'SSIM': ssim, 'Image': visulize_rgbs})
        merged_images = torchvision.utils.make_grid(
            torch.stack([r['Image'] for r in _validation_outputs[:15]]), nrow=3, padding=0
        )
        merged_psnr = np.mean([r['PSNR'] for r in _validation_outputs])
        merged_ssim = np.mean([r['SSIM'] for r in _validation_outputs])
        log_str = 'Step: {:05d} / {}, \tPSNR: {:.2f}, \tSSIM: {:.4f}.'.format(
            iter_idx, self._total_iters, merged_psnr, merged_ssim, 
        )
        self.logger.debug(log_str)
        if save_ckpt:
            self._save_validation(iter_idx, merged_ssim, merged_images, log_str, larger_best=True)
        del _validation_outputs

    def _save_checkpoints(self, name='latest.pt', optimizer=False):
        if self._debug:
            return
        saving_path = os.path.join(self._dump_dir, 'checkpoints')
        # remove old best model
        if name.startswith('best'):
            models = os.listdir(saving_path)
            for m in models:
                if m.startswith('best'):
                    os.remove(os.path.join(saving_path, m))
        state = {'model': self.model, 'meta_cfg': self._meta_cfg._dump}
        if optimizer:
            state['optimizer'] = self.optimizer
        self.lightning_fabric.save(os.path.join(saving_path, name), state)

    def _save_validation(self, iter_idx, metric, images, log_string, larger_best=True):
        if self._debug:
            validation_path = os.path.join(self._dump_dir, 'debug.jpg')
        else:
            validation_path = os.path.join(self._dump_dir, 'examples', f'{iter_idx}.jpg')
        torchvision.utils.save_image(images, validation_path)
        best_path = 'best_{}_{:.3f}.pt'.format(iter_idx, metric)
        if self._best_metric is None:
            self._best_metric = metric
            self._save_checkpoints(best_path)
        else:
            if larger_best:
                if metric >= self._best_metric:
                    self._best_metric = metric
                    self._save_checkpoints(best_path)
            else:
                if metric <= self._best_metric:
                    self._best_metric = metric
                    self._save_checkpoints(best_path)

    def _logger(self, iter_idx, fit_bar, loss_metrics, show_metric):
        if not hasattr(self, 'log_stats'):
            self.log_stats, self.show_stats = [], []
        # build fit bar and file log
        learning_rate = self.optimizer.param_groups[0]['lr']
        loss_metrics = torch.utils._pytree.tree_map(lambda x: x.item(), loss_metrics)
        self.log_stats.append(loss_metrics); self.show_stats.append(show_metric)
        self.log_stats = self.log_stats[-100:]; self.show_stats = self.show_stats[-100:]
        show_metric = self._dict_mean(self.show_stats)
        show_loss = sum([float(loss_metrics[k]) for k in loss_metrics])
        fit_bar.set_postfix({'loss': "{:.4f}".format(show_loss), **show_metric})
        if iter_idx % self._log_interval == 0:
            log_metric = self._dict_mean(self.log_stats, "{:.4f}")
            log_loss = sum([float(log_metric[k]) for k in log_metric])
            log_psnr = float(show_metric['psnr'])
            log_string =  "{:05d} / {}: ".format(iter_idx, self._total_iters) + \
                            "lr={:.5f}, loss={:.4f}, psnr={:.2f} | ".format(learning_rate, log_loss, log_psnr) + \
                            ", ".join([f'{k}={v}' for k, v in log_metric.items()])
            if self._debug:
                self.logger.info(log_string)
            else:
                self.logger.debug(log_string)

    @staticmethod
    def _resize(frames, tgt_size=(256, 256)):
        if isinstance(tgt_size, torch.Tensor):
            tgt_size = (tgt_size.shape[-2], tgt_size.shape[-1])
        if frames.shape[-2:] == tgt_size:
            return frames
        else:
            frames = torchvision.transforms.functional.resize(
                frames, tgt_size, antialias=True
            )
            return frames

    @staticmethod
    def _dict_mean(dict_list, float_format='{:.2f}'):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = float_format.format(np.mean([d[key] for d in dict_list]))
        return mean_dict


if __name__ == "__main__":
    # import warnings
    # from tqdm.std import TqdmExperimentalWarning
    # warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--basemodel', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    # launch
    torch.set_float32_matmul_precision('high')
    train(args.config, args.dataset, args.basemodel, args.devices, args.debug)
