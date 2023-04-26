#  MI License

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
# modified by Igor Krawczuk and Justin Deschaneaux for research at LIONS lab at EPFL

import sys
from datetime import datetime
from typing import Optional
import os
from contextlib import nullcontext

import torch
import torch as pt
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch_fidelity
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import time
import torchvision
import numpy as np
import argparse
import os
import json

import csv

from tqdm import tqdm

import models
import utils
from utils import TensorDataset
from optim import ExtraAdam
from torch.optim import SGD,Adam
from torch import Tensor

parser = argparse.ArgumentParser()
# Original arguments
parser.add_argument('--log-path', default='results')
parser.add_argument('--model', choices=('resnet', 'dcgan'), default='resnet')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--optim', choices=['extra', 'adam', 'sgd'], default='extra',
                    help='Optimizers to use (SGD for SGDA)')
parser.add_argument('-bs', '--batch-size', default=64, type=int)
parser.add_argument('-sbs', '--score-batch-size', default=200, type=int)
parser.add_argument('--num-iter', default=500_000, type=int)
parser.add_argument('-lrd', '--learning-rate-dis', default=5e-4, type=float)
parser.add_argument('-lrg', '--learning-rate-gen', default=5e-5, type=float)
parser.add_argument('-b1', '--beta1', default=0.5, type=float)
parser.add_argument('-b2', '--beta2', default=0.9, type=float)
parser.add_argument('-ema', default=0.9999, type=float)
parser.add_argument('-nz', '--num-latent', default=128, type=int)
parser.add_argument('-nfd', '--num-filters-dis', default=128, type=int)
parser.add_argument('-nfg', '--num-filters-gen', default=128, type=int)
parser.add_argument('-gp', '--gradient-penalty', default=10, type=float)
parser.add_argument('-m', '--mode', choices=('gan','ns-gan', 'wgan'), default='wgan')
parser.add_argument('-c', '--clip', default=0.01, type=float)
parser.add_argument('-d', '--distribution', choices=('normal', 'uniform'), default='normal')
parser.add_argument('--batchnorm-dis', action='store_true')
parser.add_argument('--batchnorm-sync', action='store_true',
                    help="Replace BatchNorm modules by SyncBatchNorm in distributed training. "
                         "Exclusive with --layernorm")
parser.add_argument('--layernorm', action='store_true',
                    help='Replace batch norm. by layer norm. '
                         'Exclusive with --batchnorm-sync')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--single-gpu-force', action='store_true')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--logging-rank', type=int, default=0,
                    help="Which rank (node/gpu) should be evaluation & logging things")
parser.add_argument('--save-gen-samples', action='store_true')
parser.add_argument('--fid-score', action='store_true')
parser.add_argument('--inception-score', action='store_true')
parser.add_argument("--master-addr", default='127.0.0.1')
parser.add_argument('--score-every', type=int, default=5,
                    help="How many epochs between the FID/inception scores logging (requires their respective flags)")
parser.add_argument('--default', action='store_true')
parser.add_argument('--mpi-flag', action='store_true')

# DDP/GCX arguments
parser.add_argument('--dist-backend', choices=['cgx', 'nccl', 'gloo'], default='nccl',
                    help='Backend for torch distributed')
parser.add_argument('--gpu-count', default=1, type=int,
                    help='Number of GPUs to use for standard torch DDP (NOT GCX)'
                    )
parser.add_argument('--quantization-bits', type=int, default=32,
                    help='Quantization bits for cgx')
parser.add_argument('--quantization-bucket-size', type=int, default=1024,
                    help='Bucket size for quantization in cgx')
parser.add_argument('--local-rank', type=int, default=-1,
                    help='Local rank in distributed launch')
parser.add_argument('--num-threads', type=int, default=4,
                    help='Limits the number of CPU threads to be used per worker')
# Checkpoint loading
parser.add_argument('--wandb-checkpoint', type=Optional[str], default=None,
                    help='Weights and Biases checkpoint path, as given in the artifact/files section of runs')

args = parser.parse_args()


# =============================== INITIALIZE TORCH_CGX ===============================
def init_cgx(args):
    assert "OMPI_COMM_WORLD_SIZE" in os.environ, "Launch with mpirun"
    import torch_cgx
    if 'CGX_COMPRESSION_QUANTIZATION_BITS' not in os.environ:
        print(f"\nUsing {args.quantization_bits} quantization bits\n")
        os.environ['CGX_COMPRESSION_QUANTIZATION_BITS'] = str(args.quantization_bits)
    if 'CGX_COMPRESSION_BUCKET_SIZE' not in os.environ:
        os.environ['CGX_COMPRESSION_BUCKET_SIZE'] = str(args.quantization_bucket_size)

    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = '4040'
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = world_size > 1
        args.local_rank = args.local_rank % torch.cuda.device_count()
        dist.init_process_group('cgx', init_method="env://", rank=args.local_rank)
        args.world_size = torch.distributed.get_world_size()
    print("CGX Successfully set up")
def init_mpi_nccl(args):
    assert "OMPI_COMM_WORLD_SIZE" in os.environ, "Launch with mpirun"
    if 'CGX_COMPRESSION_QUANTIZATION_BITS' not in os.environ:
        print(f"\nUsing {args.quantization_bits} quantization bits\n")
        os.environ['CGX_COMPRESSION_QUANTIZATION_BITS'] = str(args.quantization_bits)
    if 'CGX_COMPRESSION_BUCKET_SIZE' not in os.environ:
        os.environ['CGX_COMPRESSION_BUCKET_SIZE'] = str(args.quantization_bucket_size)

    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = '4040'
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = world_size > 1
        args.local_rank = args.local_rank % torch.cuda.device_count()
        dist.init_process_group('mpi', rank=0,world_size=0)
        args.world_size = torch.distributed.get_world_size()
    print("mpi-nccl Successfully set up")


def register_layers(model):
    import torch_cgx
    layers = [(name, p.numel()) for name, p in model.named_parameters()]
    torch_cgx.register_model(layers)
    torch_cgx.exclude_layer("bn")  # all batch norm layers
    torch_cgx.exclude_layer("bias")  # all bias modules

# ====================================================================================


# =============================== INITIALIZE TORCH DDP (NOT CGX) ===============================
def torch_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(args.dist_backend, rank=rank, world_size=world_size)
    training(rank, world_size, args)

# ==============================================================================================


def training(rank, world_size, args):
    print(f"Process with rank {rank} spawned")

    CUDA = args.cuda
    MODEL = args.model
    GRADIENT_PENALTY = args.gradient_penalty
    OUTPUT_PATH = args.log_path
    TENSORBOARD_FLAG = args.tensorboard
    WANDB_FLAG = args.wandb
    WANDB_CHECKPOINT = args.wandb_checkpoint

    INCEPTION_SCORE_FLAG = args.inception_score
    FID_SCORE_FLAG = args.fid_score
    SCORE_EVERY = args.score_every

    if args.default:
        if args.model == 'resnet' and args.gradient_penalty != 0:
            config = "config/default_resnet_wgangp_extraadam.json"
        elif args.model == 'dcgan' and args.gradient_penalty != 0:
            config = "config/default_dcgan_wgangp_extraadam.json"
        elif args.model == 'dcgan' and args.gradient_penalty == 0:
            config = "config/default_dcgan_wgan_extraadam.json"
        else:
            raise ValueError("Not default config available for this.")
        with open(config) as f:
            config_params = json.load(f)
            cli_params = vars(args)
            merged_params = {**config_params, **cli_params}
        args = argparse.Namespace(**merged_params)

    BATCH_SIZE = args.batch_size // world_size
    N_ITER = args.num_iter
    LEARNING_RATE_G = args.learning_rate_gen # It is really important to set different learning rates for the discriminator and generator
    LEARNING_RATE_D = args.learning_rate_dis
    BETA_1 = args.beta1
    BETA_2 = args.beta2
    BETA_EMA = args.ema
    N_LATENT = args.num_latent
    N_FILTERS_G = args.num_filters_gen
    N_FILTERS_D = args.num_filters_dis
    MODE = args.mode
    CLIP = args.clip
    DISTRIBUTION = args.distribution
    BATCH_NORM_G = True
    BATCH_NORM_D = args.batchnorm_dis
    N_SAMPLES = 50000
    RESOLUTION = 32
    N_CHANNEL = 3
    EVAL_FREQ = 20_000
    SEED = args.seed
    EXTRA=args.optim=="extra"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_gen_update = 0
    n_dis_update = 0
    total_time = 0

    torch.set_num_threads(args.num_threads)
    now = datetime.now()
    current_time = now.strftime("%d/%m/%y %H:%M:%S")
    variant = "extra_adam" + ("-gp" if GRADIENT_PENALTY else "")

    MODEL_NAME = f"{current_time}-{MODEL}_{MODE}-{variant}"
    ARTIFACT_NAME = f"{now.strftime('%d-%m-%y--%H-%M-%S')}-{MODEL}_{MODE}-{variant}"

    if args.cuda:
        # Optimize run for current hardware. Note: needs inputs of fixed size
        cudnn.benchmark = True
        # assume we manage gpus outside and default to cuda:0 here if this is set
        if not args.single_gpu_force:
            torch.cuda.set_device(rank)
        torch.cuda.manual_seed(args.seed)

    OUTPUT_PATH = os.path.join(
        OUTPUT_PATH,
        f"{MODEL}_{MODE}" + ("-gp" if GRADIENT_PENALTY else ""),
        "extra_adam",
        f"lrd={LEARNING_RATE_D:.1e}_lrg={LEARNING_RATE_G:.1e}",
        str(SEED),
        current_time
    )

    if TENSORBOARD_FLAG and rank == args.logging_rank:
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(OUTPUT_PATH, 'tensorboard'))
        tensorboard_writer.add_text('config', json.dumps(vars(args), indent=2, sort_keys=True))

    if WANDB_FLAG and rank == args.logging_rank:
        run = wandb.init(
            project="quantized-gen-extra-grad",
            config=args
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    train_dataloader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Use sampler only in distributed settings
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        shuffle=world_size == 1,
        **train_dataloader_kwargs
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        num_workers=1,
    )

    if rank == args.logging_rank:
        print('Init....')
        if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
            os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))
        if not os.path.exists(os.path.join(OUTPUT_PATH, 'gen')):
            os.makedirs(os.path.join(OUTPUT_PATH, 'gen'))

        print(f"Model: {MODEL_NAME}")
        print("Arguments:")
        for k, v in vars(args).items():
            print("\t", k, ":", v)

    def gen_fake_samples(model, batch_size=100):
        all_samples = []
        model.eval()
        with torch.no_grad():
            for idx in range(0, N_SAMPLES, batch_size):
                samples_batch = torch.randn(batch_size, N_LATENT)
                if CUDA:
                    samples_batch = samples_batch.cuda()
                all_samples.append(model(samples_batch).cpu())
        all_samples = torch.cat(all_samples, dim=0)
        # [-1, +1] => [0, 255]
        all_samples = (255 * (all_samples.clamp(-1, 1) * 0.5 + 0.5))
        all_samples = all_samples.to(torch.uint8)
        model.train()
        return all_samples

    def log(args, step):
        if WANDB_FLAG:
            wandb.log(args)
        if TENSORBOARD_FLAG:
            for k, v in args.values():
                if isinstance(v, (float, int)):
                    tensorboard_writer.add_scalar(k, v, step)

    # Create tensor of real samples for FID or ISC computation during evaluation
    if (FID_SCORE_FLAG or INCEPTION_SCORE_FLAG) and rank == args.logging_rank:
        real_samples = []
        local_trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
        )

        for x in local_trainloader:
            batch, _ = x
            real_samples.append(batch.cpu())
        real_samples = torch.cat(real_samples, axis=0)
        real_samples = (255 * (real_samples.clamp(-1, 1) * 0.5 + 0.5))
        real_samples = real_samples.to(torch.uint8)
        real_samples_dataset = TensorDataset(real_samples)

    if MODEL == "resnet":
        gen = models.ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
        dis = models.ResNet32Discriminator(N_CHANNEL, 1, N_FILTERS_D, BATCH_NORM_D)
    elif MODEL == "dcgan":
        gen = models.DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)
        dis = models.DCGAN32Discriminator(N_CHANNEL, 1, N_FILTERS_D, batchnorm=BATCH_NORM_D)
    else:
        raise ValueError(f"Unknown model {MODEL}")

    assert not (args.layernorm and args.batchnorm_sync)

    if args.layernorm:
        gen = utils.batchnorm_to_layernorm(gen)
        dis = utils.batchnorm_to_layernorm(dis)

    gen.apply(lambda x: utils.weight_init(x, mode='normal'))
    dis.apply(lambda x: utils.weight_init(x, mode='normal'))

    if CUDA:
        gen = gen.cuda()
        dis = dis.cuda()

    if world_size > 1:
        if args.batchnorm_sync:
            gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
            dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dis)
        gen = DDP(gen, device_ids=[rank if not args.single_gpu_force else 0])
        dis = DDP(dis, device_ids=[rank if not args.single_gpu_force else 0])

    if args.optim=="extra":
        dis_optimizer = ExtraAdam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
        gen_optimizer = ExtraAdam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
    elif args.optim=="sgd":
        dis_optimizer = SGD(dis.parameters(), lr=LEARNING_RATE_D)
        gen_optimizer = SGD(gen.parameters(), lr=LEARNING_RATE_G)
    elif args.optim=="adam":
        dis_optimizer = Adam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
        gen_optimizer = Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
    else:
        raise NotImplementedError(f"Don't know {args.optim=}")

    if rank == args.logging_rank:
        with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
            json.dump(vars(args), f)

    dataiter = iter(testloader)
    examples, labels = next(dataiter)
    if rank == args.logging_rank:
        torchvision.utils.save_image(utils.unormalize(examples), os.path.join(OUTPUT_PATH, 'examples.png'))

    z_examples = utils.sample(DISTRIBUTION, (100, N_LATENT))
    if CUDA:
        z_examples = z_examples.cuda()

    gen_param_avg = []
    gen_param_ema = []
    for param in gen.parameters():
        gen_param_avg.append(param.data.clone())
        gen_param_ema.append(param.data.clone())

    if rank == args.logging_rank:
        f = open(os.path.join(OUTPUT_PATH, 'results.csv'), 'a')
        f_writter = csv.writer(f)

    n_iteration_t = 0
    epoch = 0

    if WANDB_CHECKPOINT is not None:
        assert WANDB_FLAG, "please use --wandb flag to load checkpoint"
        artifact = run.use_artifact(WANDB_CHECKPOINT, type='model')
        artifact_dir = artifact.download()
        state_file = [f for f in os.listdir(artifact_dir)][0]
        file_path = os.path.join(artifact_dir, state_file)
        state = torch.load(file_path)

        n_gen_update = state['n_gen_update']
        n_dis_update = state['n_dis_update']
        epoch = state['epoch']
        total_time = state['total_time']
        gen.load_state_dict(state['state_gen'])
        dis.load_state_dict(state['state_dis'])
        gen_optimizer.load_state_dict(state['state_gen_opt'])
        dis_optimizer.load_state_dict(state['state_dis_opt'])
        gen_param_avg = state['gen_param_avg']
        gen_param_ema = state['gen_param_ema']

        if CUDA:
            gen = gen.cuda()
            dis = dis.cuda()

    if rank == args.logging_rank:
        pbar = tqdm(total=N_ITER, desc="Training...")
    original_n_gen_update = n_gen_update
    # In case we load from a checkpoint
    gen_loss_bt=0
    dis_loss_bt=0
    penalty_bt=0
    while n_gen_update < N_ITER + original_n_gen_update:
        t = time.time()
        avg_loss_G = 0
        avg_loss_D = 0
        avg_penalty = 0
        num_samples = 0
        penalty = torch.tensor([0.], requires_grad=True)
        if CUDA:
            penalty = penalty.cuda()
        for i, data in enumerate(trainloader):
            extrapolation = (n_iteration_t+1)%2 != 0
            gen.train()
            dis.train()
            _t = time.time()
            x_true, _ = data

            z = utils.sample(DISTRIBUTION, (len(x_true), N_LATENT))
            z.requires_grad = True
            if CUDA:
                x_true = x_true.cuda()
                z = z.cuda()

            x_gen = gen(z)
            p_true, p_gen = dis(x_true), dis(x_gen)

            assert MODE=="wgan"
            gen_score=p_gen.mean()
            true_score=p_true.mean()
            gen_loss = -gen_score
            dis_loss = - (true_score-gen_score)
            if GRADIENT_PENALTY:
                penalty_t0=time.time()
                penalty = (dis.module if world_size > 1 else dis).get_penalty(x_true.data, x_gen.data)
                penalty_bt+=time.time()-penalty_t0
                dis_loss = dis_loss+ GRADIENT_PENALTY*penalty

            for p in gen.parameters():
                p.requires_grad = False
            dis_optimizer.zero_grad()
            dis_loss_backward_t0=time.time()
            dis_loss.backward(retain_graph=True) # since we call gen_loss_backward right after in 611
            dis_loss_bt+=-dis_loss_backward_t0+time.time()


            for p in gen.parameters():
                p.requires_grad = True

            for p in dis.parameters():
                p.requires_grad = False
            gen_optimizer.zero_grad()
            gen_loss_backward_t0=time.time()
            gen_loss.backward()
            gen_loss_bt+=-gen_loss_backward_t0+time.time()


            if  EXTRA and extrapolation:
                dis_optimizer.extrapolation()
            else:
                n_dis_update += 1
                dis_optimizer.step()

            if EXTRA and extrapolation:
                gen_optimizer.extrapolation()
            else:
                n_gen_update += 1
                gen_optimizer.step()
                for j, param in enumerate(gen.parameters()):
                    gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
                    gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)

            for p in dis.parameters():
                p.requires_grad = True

            if MODE == 'wgan' and not GRADIENT_PENALTY:
                for p in dis.parameters():
                    p.data.clamp_(-CLIP, CLIP)

            total_time += time.time() - _t
            if (n_iteration_t+1) % 2 == 0:

                avg_loss_D += dis_loss.item()*len(x_true)
                avg_loss_G += gen_loss.item()*len(x_true)
                avg_penalty += penalty.item()*len(x_true)
                num_samples += len(x_true)

            if rank == args.logging_rank:
                pbar.update()
                if n_gen_update % EVAL_FREQ == 1:
                    print("\n checkpointing \n")
                    checkpoint_path = os.path.join(OUTPUT_PATH, "checkpoints", "%i.state" % n_gen_update)
                    torch.save({
                        'args': vars(args),
                        'n_gen_update': n_gen_update,
                        'n_dis_update': n_dis_update,
                        'epoch': epoch,
                        'total_time': total_time,
                        'state_gen': (gen.module if world_size > 1 else gen).state_dict(),
                        'state_dis': (dis.module if world_size > 1 else dis).state_dict(),
                        'state_gen_opt': gen_optimizer.state_dict(),
                        'state_dis_opt': dis_optimizer.state_dict(),
                        'gen_param_avg': gen_param_avg,
                        'gen_param_ema': gen_param_ema},
                        checkpoint_path
                    )

                    if WANDB_FLAG:
                        checkpoint_artifact = wandb.Artifact(
                            name=ARTIFACT_NAME,
                            type="model",
                        )
                        checkpoint_artifact.add_file(checkpoint_path)
                        wandb.log_artifact(checkpoint_artifact)

            n_iteration_t += 1

        epoch += 1
        avg_loss_G /= num_samples
        avg_loss_D /= num_samples
        avg_penalty /= num_samples

        if rank == 0:
            print('Iter: %i, Loss Generator: %.4f, Loss Discriminator: %.4f, Penalty: %.2e, Time: %.4f'
                  % (n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, time.time() - t))

            f_writter.writerow((n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, time.time() - t))
            f.flush()

            x_gen = gen(z_examples)
            x = utils.unormalize(x_gen)
            torchvision.utils.save_image(x.data, os.path.join(OUTPUT_PATH, 'gen/%i.png' % n_gen_update))
            x = torchvision.utils.make_grid(x.data, 10)
            log_dict = {
                "loss_G": avg_loss_G,
                "loss_D": avg_loss_D,
                "penalty": avg_penalty,
            }

            if args.save_gen_samples:
                log_dict[f"gen-{n_gen_update}"] = wandb.Image(x)

            log(log_dict, n_gen_update)

        # ========================== MODEL EVALUATION: INCEP. SCORE + FID ==========================
        if rank == args.logging_rank:
            to_log = dict()
            if (FID_SCORE_FLAG or INCEPTION_SCORE_FLAG) \
                    and epoch % SCORE_EVERY == 1:
                fake_samples = gen_fake_samples(gen.module if world_size > 1 else gen,batch_size=args.score_batch_size)
                print("\nComputing scores\n")
                # Wrap the generator for torch fidelity
                wrapper = utils.PostProcessingWrapper(gen)
                wrapper = torch_fidelity.GenerativeModelModuleWrapper(
                    wrapper,
                    N_LATENT,
                    DISTRIBUTION,
                    0
                )
                gen.eval()
                t = time.time()
                metrics = torch_fidelity.calculate_metrics(
                    input1=wrapper,
                    input2='cifar10-train',
                    input1_model_num_samples=50_000,
                    input2_model_num_samples=50_000,
                    cuda=CUDA,
                    isc=INCEPTION_SCORE_FLAG,
                    fid=FID_SCORE_FLAG,
                    kid=False,
                    verbose=True,
                    datasets_root='./data',
                )
                score_computation_time = time.time() - t
                to_log = {
                    "ISC_mean": metrics['inception_score_mean'],
                    "ISC_std": metrics['inception_score_std'],
                    "FID": metrics['frechet_inception_distance'],
                    "scores_comp_time": score_computation_time,
                }
                print("\nScores computed!\n")
            to_log["total_time"] = total_time
            to_log["penalty_bt"]=penalty_bt
            to_log["dis_loss_bt"]=dis_loss_bt
            to_log["gen_loss_bt"]=gen_loss_bt
            log(to_log, n_gen_update)

    if rank == args.logging_rank:
        pbar.close()


def main(args):
    ctx=torch.autograd.detect_anomaly if os.environ.get("ANOMALY",None) is not None else nullcontext
    if args.dist_backend == "cgx":
        init_cgx(args)
        world_size = int(os.environ["WORLD_SIZE"])
        assert 0 <= args.logging_rank < world_size, \
            f"Invalid logging rank: {args.logging_rank} with world size {world_size}"
        rank = int(os.environ["RANK"])
        with ctx():
            training(rank, world_size, args)

    elif args.dist_backend in ("nccl", "gloo"):
        world_size = args.gpu_count
        assert 0 <= args.logging_rank < world_size, \
            f"Invalid logging rank: {args.logging_rank} with world size {world_size}"
        if args.mpi_flag:
            init_mpi_nccl(args)
            world_size = int(os.environ["WORLD_SIZE"])
            assert 0 <= args.logging_rank < world_size, \
                f"Invalid logging rank: {args.logging_rank} with world size {world_size}"
            rank = int(os.environ["RANK"])
            with ctx():
                training(rank, world_size, args)
        elif world_size == 1:
            with ctx():
                training(0, 1, args)
        else:
            torch.multiprocessing.spawn(
                torch_ddp,
                args=(world_size, args),
                nprocs=world_size
            )
    else:
        raise ValueError(f"Unknown backend {args.dist_backend}")


if __name__ == "__main__":
    main(args)
