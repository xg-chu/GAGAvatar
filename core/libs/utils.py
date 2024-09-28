#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import time
import yaml
import random
import logging
from functools import partial
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import colored
import matplotlib
import numpy as np
import torchvision
import torchmetrics
from omegaconf import OmegaConf
from fvcore.common.registry import Registry

DATA_REGISTRY = Registry("DATA")
MODEL_REGISTRY = Registry("MODEL")
calc_psnr = torchmetrics.functional.image.peak_signal_noise_ratio

def calc_ssim(preds, target, data_range=None):
    preds_gray = torchvision.transforms.functional.rgb_to_grayscale(preds)
    target_gray = torchvision.transforms.functional.rgb_to_grayscale(target)
    ssim = torchmetrics.functional.image.structural_similarity_index_measure(
        preds_gray, target_gray, data_range=data_range,
    )
    return ssim


def device_parser(str_device):
    def parser_dash(str_device):
        device_id = str_device.split('-')
        device_id = [i for i in range(int(device_id[0]), int(device_id[-1])+1)]
        return device_id
    if 'cpu' in str_device:
        device_id = ['cpu']
    else:
        device_id = str_device.split(',')
        device_id = [parser_dash(i) for i in device_id]
    res = []
    for i in device_id:
        res += i
    return res


def pretty_dict(input_dict, indent=0, highlight_keys=[]):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        if key in highlight_keys:
            out_line += tab * indent + colored.stylize(str(key), colored.fg(1))
        else:
            out_line += tab * indent + colored.stylize(str(key), colored.fg(2))
        if isinstance(value, dict):
            out_line += ':\n'
            out_line += pretty_dict(value, indent+1, highlight_keys)
        else:
            if key in highlight_keys:
                out_line += ":" + "\t" + colored.stylize(str(value), colored.fg(1)) + '\n'
            else:
                out_line += ":" + "\t" + colored.stylize(str(value), colored.fg(2)) + '\n'
    if indent == 0:
        max_length = 0
        for line in out_line.split('\n'):
            max_length = max(max_length, len(line.split('\t')[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split('\n'):
            if '\t' in line:
                aligned_number = max_length - len(line.split('\t')[0])
                line = line.replace('\t',  aligned_number * ' ')
            aligned_line += line+'\n'
        return aligned_line[:-2]
    return out_line


def get_time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def biuld_logger(log_path, name='test_logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def vis_depth(depth_map, color_map='magma', non_linear=False):
    # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    def percentile(x, ps, assume_sorted=False):
        """Compute the percentile(s) of a single vector."""
        x = x.reshape([-1])
        if not assume_sorted:
            x = np.sort(x)
        acc = np.arange(1, x.shape[0] + 1) / x.shape[0]
        return np.interp(np.array(ps) / 100, acc, x)
    if non_linear:
        depth_curve_fn = lambda x: (x / 2.5 + 1) ** -1.5
        depth_map = depth_curve_fn(depth_map)
        # lo_auto, hi_auto = percentile(
        #     depth_map, [0.1, 99.9], assume_sorted=True
        # )
        # depth_map = (depth_map - min(lo_auto, hi_auto)) / np.abs(hi_auto - lo_auto)
        # depth_map = depth_map.clamp(0, 1)
        depth_map = depth_map - depth_map.min()
        depth_map = depth_map / depth_map.max()
    else:
        depth_map = depth_map - depth_map.min()
        depth_map = depth_map / depth_map.max()
    original_dim = depth_map.dim()
    assert original_dim in [2, 3], depth_map.dim()
    if original_dim == 2:
        depth_map = depth_map[None]
    device = depth_map.device
    depth_color = matplotlib.colormaps[color_map]
    depth_map = torch.tensor(
        depth_color(depth_map.cpu().numpy())[..., :3], device=device
    ).permute(0, 3, 1, 2)
    if original_dim == 2:
        depth_map = depth_map[0]
    return depth_map


def find_best_model(dir_path):
    models = os.listdir(dir_path)
    model_path = models[0]
    for m in models:
        if 'best' in m:
            model_path = m
    return model_path


def calc_parameters(model):
    op_para_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_para_num = sum(p.numel() for p in model.parameters())
    return op_para_num, all_para_num


class SphericalHarmonics(torch.nn.Module):
    def __init__(self, degree=4):
        super().__init__()
        self.degree = max([int(degree), 0])
        self.out_dim = self.degree * self.degree

    def Lengdre_polynormial(self, x, omx=None):
        if omx is None: omx = 1 - x * x
        Fml = [[]] * ((self.degree + 1) * self.degree // 2)
        Fml[0] = torch.ones_like(x)
        for l in range(1, self.degree):
            b = (l * l + l) // 2
            Fml[b + l] = -Fml[b - 1] * (2 * l - 1)
            Fml[b + l - 1] = Fml[b - 1] * (2 * l - 1) * x
            for m in range(l, 1, -1):
                Fml[b + m - 2] = -(omx * Fml[b + m] + \
                                   2 * (m - 1) * x * Fml[b + m - 1]) / ((l - m + 2) * (l + m - 1))
        return Fml

    def SH(self, xyz):
        cs = xyz[..., 0:1]
        sn = xyz[..., 1:2]
        Fml = self.Lengdre_polynormial(xyz[..., 2:3], cs * cs + sn * sn)
        H = [[]] * (self.degree * self.degree)
        for l in range(self.degree):
            b = l * l + l
            attr = np.sqrt((2 * l + 1) / math.pi / 4)
            H[b] = attr * Fml[b // 2]
            attr = attr * np.sqrt(2)
            snM = sn
            csM = cs
            for m in range(1, l + 1):
                attr = -attr / np.sqrt((l + m) * (l + 1 - m))
                H[b - m] = attr * Fml[b // 2 + m] * snM
                H[b + m] = attr * Fml[b // 2 - m] * csM
                snM, csM = snM * cs + csM * sn, csM * cs - snM * sn
        if len(H) > 0:
            return torch.cat(H, -1)
        else:
            return torch.Tensor([])

    def forward(self, coords):
        assert coords.shape[-1] == 3
        return self.SH(coords)


class SHEncoding(torch.nn.Module):
    """Spherical harmonic encoding
    Args:
        levels: Number of spherical harmonic levels to encode.
    """
    def __init__(self, degree: int = 4, ) -> None:
        super().__init__()
        if degree <= 0 or degree > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {degree}")
        self.levels = degree
        self.out_dim = self.levels**2

    def forward(self, directions):
        assert directions.shape[-1] == 3, f"Direction should have three dimensions: {directions.shape}"
        num_components = self.levels**2
        components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

        x = directions[..., 0]
        y = directions[..., 1]
        z = directions[..., 2]

        xx = x**2
        yy = y**2
        zz = z**2

        # l0
        components[..., 0] = 0.28209479177387814

        # l1
        if self.levels > 1:
            components[..., 1] = 0.4886025119029199 * y
            components[..., 2] = 0.4886025119029199 * z
            components[..., 3] = 0.4886025119029199 * x

        # l2
        if self.levels > 2:
            components[..., 4] = 1.0925484305920792 * x * y
            components[..., 5] = 1.0925484305920792 * y * z
            components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
            components[..., 7] = 1.0925484305920792 * x * z
            components[..., 8] = 0.5462742152960396 * (xx - yy)

        # l3
        if self.levels > 3:
            components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
            components[..., 10] = 2.890611442640554 * x * y * z
            components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
            components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
            components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
            components[..., 14] = 1.445305721320277 * z * (xx - yy)
            components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

        # l4
        if self.levels > 4:
            components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
            components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
            components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
            components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
            components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
            components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
            components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
            components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
            components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return components


def positional_encoding(x, min_deg=0, max_deg=4, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg, device=x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat



def correct_color(img, ref, num_iters=5, eps=0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    def matmul(a, b):
        # B,3,4,1  B,1,4,3
        # torch.matmul(a, b) cause nan when fp16
        return (a[..., None] * b[..., None, :, :]).sum(dim=-2)
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(mask, b, torch.zeros_like(b))
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img


### RICH TQDM ###
from tqdm.std import tqdm as std_tqdm
from rich.progress import BarColumn, Progress, ProgressColumn, Text, TimeElapsedColumn, TimeRemainingColumn, filesize
class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""
    def __init__(self, unit_scale=False, unit_divisor=1000):
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Calculate common unit for completed and total."""
        completed = int(task.completed)
        total = int(task.total)
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                total,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(total, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(
            f"{completed/unit:,.{precision}f}/{total/unit:,.{precision}f} {suffix}",
            style="progress.download")


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""
    def __init__(self, unit="", unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(f"{speed/unit:,.{precision}f} {suffix}{self.unit}/s",
                    style="progress.data.speed")


class rtqdm(std_tqdm):  # pragma: no cover
    """Experimental rich.progress GUI version of tqdm!"""
    # TODO: @classmethod: write()?
    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        options  : dict, optional
            keyword arguments for `rich.progress.Progress()`.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        progress = kwargs.pop('progress', None)
        options = kwargs.pop('options', {}).copy()
        super(rtqdm, self).__init__(*args, **kwargs)

        if self.disable:
            return

        # warn("rich is experimental/alpha", TqdmExperimentalWarning, stacklevel=2)
        d = self.format_dict
        if progress is None:
            progress = (
                "[progress.description]"
                "[progress.percentage]{task.percentage:>4.0f}%",
                BarColumn(bar_width=66),
                FractionColumn(unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                "[", 
                    TimeElapsedColumn(), "<", TimeRemainingColumn(), ",", 
                    RateColumn(unit=d['unit'], unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                    "{task.description}",
                "]", 
            )
        options.setdefault('transient', not self.leave)
        self._prog = Progress(*progress, **options)
        self._prog.__enter__()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable:
            return
        super(rtqdm, self).close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        pass

    def set_postfix(self, desc):
        import colored
        desc_str = ", "+" , ".join([
            colored.stylize(str(f"{k}"), colored.fg(3)) + " = " +
            colored.stylize(str(f"{v}"), colored.fg(4))
            for k, v in desc.items()]
        )
        self.desc = desc_str
        self.display()

    def display(self, *_, **__):
        if not hasattr(self, '_prog'):
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(total=total)
        super(rtqdm, self).reset(total=total)


### CONFIG ###
def read_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


class ConfigDict(dict):
    def __init__(self, model_config_path=None, data_config_path=None, init_dict=None):
        if init_dict is None:
            # build new config
            config_dict = read_config(model_config_path)
            if data_config_path is not None:
                dataset_dict = read_config(data_config_path)
                merge_a_into_b(dataset_dict, config_dict)
            # set output path 
            experiment_string = '{}_{}'.format(
                config_dict['MODEL']['NAME'], config_dict['DATASET']['NAME']
            )
            timeInTokyo = datetime.now()
            timeInTokyo = timeInTokyo.astimezone(ZoneInfo('Asia/Tokyo'))
            time_string = timeInTokyo.strftime("%b%d_%H%M_")+ \
                        "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
            config_dict['TRAIN']['EXP_STR'] = experiment_string
            config_dict['TRAIN']['TIME_STR'] = time_string
        else:
            config_dict = init_dict
        super().__init__(config_dict)
        self._dot_config = OmegaConf.create(dict(self))
        OmegaConf.set_readonly(self._dot_config, True)

    def __getattr__(self, name):
        if name == '_dump':
            return dict(self)
        if name == '_raw_string':
            import re
            ansi_escape = re.compile(r'''
                \x1B  # ESC
                (?:   # 7-bit C1 Fe (except CSI)
                    [@-Z\\-_]
                |     # or [ for CSI, followed by a control sequence
                    \[
                    [0-?]*  # Parameter bytes
                    [ -/]*  # Intermediate bytes
                    [@-~]   # Final byte
                )
            ''', re.VERBOSE)
            result = '\n' + ansi_escape.sub('', pretty_dict(self))
            return result
        return getattr(self._dot_config, name)

    def __str__(self, ):
        return pretty_dict(self)

    def update(self, key, value):
        OmegaConf.set_readonly(self._dot_config, False)
        self._dot_config[key] = value
        self[key] = value
        OmegaConf.set_readonly(self._dot_config, True)


def render_points(points, E=15, A=45, color_map='magma', scale=20, alpha=0.1, no_aixs=False):
    assert points.dim() == 3 and points.shape[-1] == 3, points.shape
    import io
    import matplotlib
    from PIL import Image
    import matplotlib.pyplot as plt
    if isinstance(points, torch.Tensor):
        points = points.detach().clone().cpu().numpy()
    else:
        points = points.copy()
    if points.shape[1] >= 30000:
        points = points[:, np.random.choice(points.shape[1], 30000)]
    images = []
    for point in points:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection='3d')
        if no_aixs:
            ax.axis('off')
        ax.axes.set_xlim3d(left=-1.0-1e-16, right=1.0+1e-16)
        ax.axes.set_ylim3d(bottom=-1.0-1e-16, top=1.0+1e-16) 
        ax.axes.set_zlim3d(bottom=-1.0-1e-16, top=1.0+1e-16) 
        ax.view_init(elev=E, azim=A)
        # pytorch3d coords to matplotlib coords
        point = point[..., [0, 2, 1]]
        # set point color along z axis
        color_axis = point[:, 1].copy() * 0.5 + 0.5
        colors = matplotlib.colormaps[color_map](color_axis)
        ax.scatter(point[:, 0], point[:, 1], point[:, 2], c=colors, marker='o', s=scale, alpha=alpha)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        plt.close()
        images.append(torchvision.transforms.functional.pil_to_tensor(Image.open(buf))[:3])
    images = torch.stack(images)
    return images
