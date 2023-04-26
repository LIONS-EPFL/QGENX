#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


def clip_weights(params, clip=0.01):
    for p in params:
        p.clamp_(-clip, clip)


def unormalize(x):
    return x/2. + 0.5


def sample(name, size):
    if name == 'normal':
        return torch.zeros(size).normal_()
    elif name == 'uniform':
        return torch.zeros(size).uniform_()
    else:
        raise ValueError()


def weight_init(m, mode='normal'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        if mode == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'kaimingu':
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, 0.8)


def compute_gan_loss(p_true, p_gen, mode='gan', gen_flag=False):
    if mode == 'ns-gan' and gen_flag:
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - (p_gen.clamp(max=0) - torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'gan' or mode == 'gan++':
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - (p_gen.clamp(min=0) + torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'wgan':
        loss = p_true.mean() - p_gen.mean()
    else:
        raise NotImplementedError()

    return loss


def batchnorm_to_layernorm(module):
    # TODO: check it removes old module with same name
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        return torch.nn.GroupNorm(1, module.num_features)
    elif isinstance(module, torch.nn.BatchNorm1d):
        return torch.nn.Sequential(
            Unsqueeze(),
            torch.nn.GroupNorm(1, module.num_features),
            Squeeze()
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, batchnorm_to_layernorm(child)
        )

    # TODO: del ?
    return module_output


class TensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, item):
        return self.tensor[item]


class Unsqueeze(nn.Module):
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        return x


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        return x


class PostProcessingWrapper(nn.Module):
    def __init__(self, module):
        super(PostProcessingWrapper, self).__init__()
        self.module = module

    def forward(self, x):
        output = self.module(x)
        output = 255 * (output.clamp(-1, 1) * 0.5 + 0.5)
        return output.to(torch.uint8)
