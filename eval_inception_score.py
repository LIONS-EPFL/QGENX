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

import models
import argparse
import os
import torch
from torch.autograd import Variable
import numpy as np
import csv
import glob

parser = argparse.ArgumentParser()
parser.add_argument('input')
args = parser.parse_args()

INPUT_PATH = args.input
CUDA = True
BATCH_SIZE = 1000
N_CHANNEL = 3
RESOLUTION = 32
NUM_SAMPLES = 50000
DEVICE = 'cpu'

checkpoint = torch.load(INPUT_PATH, map_location=DEVICE)
args = argparse.Namespace(**checkpoint['args'])

MODEL = args.model
N_LATENT = args.num_latent
N_FILTERS_G = args.num_filters_gen
BATCH_NORM_G = True

print "Init..."
import tflib
import tflib.inception_score
def get_inception_score():
    all_samples = []
    samples = torch.randn(NUM_SAMPLES, N_LATENT)
    for i in xrange(0, NUM_SAMPLES, BATCH_SIZE):
        samples_100 = samples[i:i+BATCH_SIZE]
        if CUDA:
            samples_100 = samples_100.cuda(0)
        all_samples.append(gen(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, N_CHANNEL, RESOLUTION, RESOLUTION)).transpose(0, 2, 3, 1)
    return tflib.inception_score.get_inception_score(list(all_samples))

if MODEL == "resnet":
    gen = models.ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
elif MODEL == "dcgan":
    gen = models.DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)

print "Eval..."
gen.load_state_dict(checkpoint['state_gen'])
if CUDA:
    gen.cuda(0)
inception_score = get_inception_score()[0]

for j, param in enumerate(gen.parameters()):
    param.data = checkpoint['gen_param_avg'][j]
if CUDA:
    gen = gen.cuda(0)
inception_score_avg = get_inception_score()[0]

for j, param in enumerate(gen.parameters()):
    param.data = checkpoint['gen_param_ema'][j]
if CUDA:
    gen = gen.cuda(0)
inception_score_ema = get_inception_score()[0]


print 'IS: %.2f, IS Avg: %.2f, IS EMA: %.2f'%(inception_score, inception_score_avg, inception_score_ema)
