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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
from .discriminator import Discriminator

class ResBlock(nn.Module):
    def __init__(self, num_filters, resample=None, batchnorm=True, inplace=False):
        super(ResBlock, self).__init__()

        if resample == 'up':
            conv_list = [nn.ConvTranspose2d(num_filters, num_filters, 4, stride=2, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut =  nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)

        elif resample == 'down':
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)]
            self.conv_shortcut = nn.Conv2d(num_filters, num_filters, 1, stride=2)

        elif resample == None:
            conv_list = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                        nn.Conv2d(num_filters, num_filters, 3, padding=1)]
            self.conv_shortcut = None
        else:
            raise ValueError('Invalid resample value.')

        self.block = []
        for conv in conv_list:
            if batchnorm:
                self.block.append(nn.BatchNorm2d(num_filters))
            self.block.append(nn.ReLU(inplace))
            self.block.append(conv)

        self.block = nn.Sequential(*self.block)


    def forward(self, x):
        shortcut = x
        if not self.conv_shortcut is None:
            shortcut = self.conv_shortcut(x)
        return shortcut + self.block(x)

class ResNet32Generator(nn.Module):
    def __init__(self, n_in, n_out, num_filters=128, batchnorm=True):
        super(ResNet32Generator, self).__init__()
        self.num_filters = num_filters

        self.input = nn.Linear(n_in, 4*4*num_filters)
        self.network = [ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=False),
                        ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=False),
                        ResBlock(num_filters, resample='up', batchnorm=batchnorm, inplace=False)]
        if batchnorm:
            self.network.append(nn.BatchNorm2d(num_filters))
        self.network += [nn.ReLU(inplace=False),
                        nn.Conv2d(num_filters, 3, 3, padding=1),
                        nn.Tanh()]

        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        x = self.input(z).view(len(z), self.num_filters, 4, 4)
        return self.network(x)

class ResNet32Discriminator(Discriminator):
    def __init__(self, n_in, n_out, num_filters=128, batchnorm=False):
        super(ResNet32Discriminator, self).__init__()

        self.block1 = nn.Sequential(nn.Conv2d(3, num_filters, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1))

        self.shortcut1 = nn.Conv2d(3, num_filters, 1, stride=2)

        self.network = nn.Sequential(ResBlock(num_filters, resample='down', batchnorm=batchnorm),
                                    ResBlock(num_filters, resample=None, batchnorm=batchnorm),
                                    ResBlock(num_filters, resample=None, batchnorm=batchnorm),
                                    nn.ReLU())
        self.output = nn.Linear(num_filters, 1)

    def forward(self, x):
        y = self.block1(x)
        y = self.shortcut1(x) + y
        y = self.network(y).mean(-1).mean(-1)
        y = self.output(y)

        return y
