"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

This file incorporates work covered by the following copyright and  permission notice:

	Copyright (c) 2018 Ignacio Rocco

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.

Source: https://github.com/ignacio-rocco/weakalign/blob/master/model/cnn_geometric_model.py

"""
import os

import torch
import torch.nn as nn
from torchvision import models

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import math

from utils.deepcluster_vgg16 import vgg16 as deepcluster_vgg16


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = deepcluster_vgg16(sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, feature_extraction_cnn='vgg19', normalization=True, last_layer='', weights=None, use_cuda=True, gpu=0, ref_backbone=None):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        print(f"layer: {last_layer}")
        print(f"weighs: {weights}")
        # multiple extracting layers
        last_layer = last_layer.split(',')

        if feature_extraction_cnn == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                                  'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']

            start_index = 0
            self.model_list = []
            for l in last_layer:
                if l == '':
                    l = 'pool4'
                layer_idx = vgg_feature_layers.index(l)
                assert layer_idx >= start_index, 'layer order wrong!'
                model = nn.Sequential(
                    *list(self.model.features.children())[start_index:layer_idx + 1])
                self.model_list.append(model)
                start_index = layer_idx + 1

        if feature_extraction_cnn == 'vgg16_bn':
            if ref_backbone is None:
                self.model = models.vgg16_bn(pretrained=True)
            else:
                self.model = load_model(ref_backbone)
                self.model.features = torch.nn.Sequential(*list(self.model.sobel), *list(self.model.features))
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'bn1_1', 'relu1_1', 'conv1_2',  'bn1_2', 'relu1_2', 'pool1',
                                  'conv2_1', 'bn2_1', 'relu2_1', 'conv2_2', 'bn2_2', 'relu2_2', 'pool2',
                                  'conv3_1', 'bn3_1', 'relu3_1', 'conv3_2', 'bn3_2', 'relu3_2', 'conv3_3', 'bn3_3', 'relu3_3', 'pool3',
                                  'conv4_1', 'bn4_1', 'relu4_1', 'conv4_2', 'bn4_2', 'relu4_2', 'conv4_3', 'bn4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'bn5_1', 'relu5_1', 'conv5_2', 'bn5_2', 'relu5_2', 'conv5_3', 'bn5_3', 'relu5_3', 'pool5']

            start_index = 0
            self.model_list = []
            for l in last_layer:
                if l == '':
                    l = 'pool4'
                layer_idx = vgg_feature_layers.index(l)
                assert layer_idx >= start_index, 'layer order wrong!'
                model = nn.Sequential(
                    *list(self.model.features.children())[start_index:layer_idx + 1])
                self.model_list.append(model)
                start_index = layer_idx + 1
        if feature_extraction_cnn == 'vgg19':
            self.model = models.vgg19(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                                  'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']

            vgg_output_dim = [64, 64, 64, 64, 64,
                              128, 128, 128, 128, 128,
                              256, 256, 256, 256, 256, 256, 256, 256, 256,
                              512, 512, 512, 512, 512, 512, 512, 512, 512,
                              512, 512, 512, 512, 512, 512, 512, 512, 512]

            start_index = 0
            self.model_list = []
            self.out_dim = 0
            for l in last_layer:
                if l == '':
                    l = 'relu5_4'
                layer_idx = vgg_feature_layers.index(l)
                assert layer_idx >= start_index, 'layer order wrong!'
                self.out_dim += vgg_output_dim[layer_idx]
                model = nn.Sequential(
                    *list(self.model.features.children())[start_index:layer_idx + 1])
                self.model_list.append(model)
                start_index = layer_idx + 1

        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(
                *resnet_module_list[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(
                *list(self.model.features.children())[:-4])
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model_list = [model.cuda(gpu) for model in self.model_list]
        self.weights = [1]*len(self.model_list) if weights is None else weights

    def forward(self, image_batch):
        features_list = []
        features = image_batch
        for i, model in enumerate(self.model_list):
            features = model(features)
            if self.normalization:
                features = featureL2Norm(features)
            features_list.append(features * self.weights[i])
        return features_list
