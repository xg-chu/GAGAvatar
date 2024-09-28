#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import torchvision
import torch.nn as nn

class DINOBase(nn.Module):
    def __init__(self, output_dim=128, only_global=False):
        super().__init__()
        self.only_global = only_global
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.dino_normlize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if not only_global:
            in_dim = self.dino_model.blocks[0].attn.qkv.in_features
            hidden_dims=256
            out_dims=[256, 512, 1024, 1024]
            # modules
            self.projects = nn.ModuleList([
                nn.Conv2d(
                    in_dim, out_dim, kernel_size=1, stride=1, padding=0,
                ) for out_dim in out_dims
            ])
            self.resize_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    out_dims[0], out_dims[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_dims[1], out_dims[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_dims[3], out_dims[3], kernel_size=3, stride=2, padding=1
                )
            ])
            self.layer_rn = nn.ModuleList([
                nn.Conv2d(out_dims[0]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(out_dims[1]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(out_dims[2]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(out_dims[3]+3, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            ])

            self.refinenet = nn.ModuleList([
                FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
                FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
                FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
                FeatureFusionBlock(hidden_dims, nn.ReLU(False)),
            ])
            self.output_conv = nn.Conv2d(hidden_dims, output_dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, images, output_size=None):
        images = self.dino_normlize(images)
        patch_h, patch_w = images.shape[-2]//14, images.shape[-1]//14
        if self.only_global:
            image_features = self.dino_model.get_intermediate_layers(images, 1)
            out_global = image_features[0][:, 0]
            return out_global
        else:
            image_features = self.dino_model.get_intermediate_layers(images, 4)
            out_features = []
            for i, feature in enumerate(image_features):
                feature = feature.permute(0, 2, 1).reshape(
                    (feature.shape[0], feature.shape[-1], patch_h, patch_w)
                )
                feature = self.projects[i](feature)
                feature = self.resize_layers[i](feature)
                feature = torch.cat([
                        torchvision.transforms.functional.resize(images, (feature.shape[-2], feature.shape[-1]), antialias=True).detach(),
                        feature
                    ], dim=1
                )
                out_features.append(feature)
            layer_rns = []
            for i, feature in enumerate(out_features):
                layer_rns.append(self.layer_rn[i](feature))

            path_4 = self.refinenet[0](layer_rns[3], size=layer_rns[2].shape[2:])
            path_3 = self.refinenet[1](path_4, layer_rns[2], size=layer_rns[1].shape[2:])
            path_2 = self.refinenet[2](path_3, layer_rns[1], size=layer_rns[0].shape[2:])
            path_1 = self.refinenet[3](path_2, layer_rns[0])
            out = self.output_conv(path_1)
            if output_size is not None:
                out = nn.functional.interpolate(out, output_size, mode="bilinear", align_corners=True)
            out_global = image_features[-1][:, 0]
            return out, out_global


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output
