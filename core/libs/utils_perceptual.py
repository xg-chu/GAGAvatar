#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import torch.nn as nn
import torchvision

class FacePerceptualLoss(nn.Module):
    def __init__(self, loss_type='l1', weighted=True):
        super().__init__()
        self.face_layers = ["relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1"]
        self.vggface = _vgg_face(self.face_layers)
        self.loss_type = loss_type
        self.net = AlexNet()
        self.chns = [64,192,384,256,256]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if weighted:
            self.weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        else:
            self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.L = len(self.chns)
        self.norm = torchvision.transforms.Normalize(mean=self.mean, std=self.std, inplace=False)
        self.vggface.eval()
        self.eval()

    def forward(self, img0, img1, normalize=False):
        self.eval()
        self.vggface.eval()
        if normalize: # turn on this flag if input is [-1, +1] so it can be adjusted to [0, 1]
            img0 = 0.5 * img0 + 0.5
            img1 = 0.5 * img1 + 0.5
        outs0, outs1 = self.net.forward(self.norm(img0)), self.net.forward(self.norm(img1))
        features_vggface_input = self.vggface(apply_vggface_normalization(img0))
        features_vggface_target = self.vggface(apply_vggface_normalization(img1))
        final_loss, losses = 0, {}
        for lid in range(self.L):
            feats0, feats1 = self.normalize_tensor(outs0[lid]), self.normalize_tensor(outs1[lid])
            feats2, feats3 = self.normalize_tensor(features_vggface_input[self.face_layers[lid]]), self.normalize_tensor(features_vggface_target[self.face_layers[lid]])
            if self.loss_type == 'l1':
                losses[lid] = torch.nn.functional.l1_loss(feats0, feats1, reduction='none').sum(dim=1,keepdim=True).mean() * self.weights[lid]
                losses[lid] += torch.nn.functional.l1_loss(feats2, feats3, reduction='none').sum(dim=1,keepdim=True).mean() * self.weights[lid] * 0.5
            elif self.loss_type == 'l2':
                losses[lid] = torch.nn.functional.mse_loss(feats0, feats1, reduction='none').sum(dim=1,keepdim=True).mean() * self.weights[lid]
                losses[lid] += torch.nn.functional.mse_loss(feats2, feats3, reduction='none').sum(dim=1,keepdim=True).mean() * self.weights[lid] * 0.5
            else:
                raise NotImplementedError
            final_loss += losses[lid]
        return final_loss

    @staticmethod
    def normalize_tensor(in_feat,eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        return in_feat/(norm_factor+eps)


class AlexNet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        alexnet_layers = torchvision.models.alexnet(weights="DEFAULT").features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_layers[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_layers[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_layers[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_layers[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_layers[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        return (h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def _vgg_face(layers):
    network = torchvision.models.vgg16(num_classes=2622)
    state_dict = torch.utils.model_zoo.load_url(
        "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/" "vgg_face_dag.pth", map_location=torch.device("cpu"), progress=True
    )
    feature_layer_name_mapping = {
        0: "conv1_1",
        2: "conv1_2",
        5: "conv2_1",
        7: "conv2_2",
        10: "conv3_1",
        12: "conv3_2",
        14: "conv3_3",
        17: "conv4_1",
        19: "conv4_2",
        21: "conv4_3",
        24: "conv5_1",
        26: "conv5_2",
        28: "conv5_3",
    }
    new_state_dict = {}
    for k, v in feature_layer_name_mapping.items():
        new_state_dict["features." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["features." + str(k) + ".bias"] = state_dict[v + ".bias"]
    classifier_layer_name_mapping = {0: "fc6", 3: "fc7", 6: "fc8"}
    for k, v in classifier_layer_name_mapping.items():
        new_state_dict["classifier." + str(k) + ".weight"] = state_dict[v + ".weight"]
        new_state_dict["classifier." + str(k) + ".bias"] = state_dict[v + ".bias"]
    network.load_state_dict(new_state_dict)
    layer_name_mapping = {
        1: "relu_1_1",
        3: "relu_1_2",
        6: "relu_2_1",
        8: "relu_2_2",
        11: "relu_3_1",
        13: "relu_3_2",
        15: "relu_3_3",
        18: "relu_4_1",
        20: "relu_4_2",
        22: "relu_4_3",
        25: "relu_5_1",
    }
    return _PerceptualNetwork(network.features, layer_name_mapping, layers)


class _PerceptualNetwork(nn.Module):
    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def apply_vggface_normalization(input):
    mean = input.new_tensor([129.186279296875, 104.76238250732422, 93.59396362304688]).reshape(1, 3, 1, 1)
    std = input.new_tensor([1, 1, 1]).reshape(1, 3, 1, 1)
    output = (input * 255 - mean) / std
    return output

