"""
SITrainer.py — Replay + Synaptic Intelligence (SI) for continual learning in SDDAL.

Extends ContinualTrainer.py (experience replay) with SI regularisation
(Zenke et al., 2017 "Continual Learning Through Synaptic Intelligence"):

    L = QuantileLoss(low, mu, high, target)
      + (si_lambda / 2) * sum_i Omega_i * (theta_i - theta_prev_i)^2

where:
  theta_prev  = encoder weights from the END of the previous round (moving anchor)
  Omega_i     = accumulated importance of parameter i across ALL previous rounds

Omega is updated at the end of every round:
    Omega_i += clamp(running_sum_i, min=0) / (delta_i^2 + xi)
    running_sum_i -= grad_i * (theta_i - theta_prev_i)   [accumulated each batch]
    delta_i = theta_i(end of round) - theta_prev_i

Why SI over L2 and EWC for SDDAL
----------------------------------
  L2   : anchor moves each round, but NO long-term memory — encoder drifts
          gradually over 200 rounds because each round's penalty only looks
          1 round back.
  EWC  : requires a task boundary to compute Fisher; cold-start anchor traps
          the model in a bad initial state.
  SI   : anchor moves each round (like L2), but Omega ACCUMULATES across all
          rounds — parameters that have consistently mattered get a growing
          penalty, preventing long-term drift without a fixed bad anchor.

The SI state (Omega + theta_prev) is saved to {exp_path}/models/si_state.pkl
after every round and loaded at the next. Round 1 starts with Omega=0 so the
first round is effectively pure replay — expected and correct.

Usage
-----
    python3 SITrainer.py \\
        --train_data  Design_rec_1_si \\
        --pth_name    Design_rec_1_si/models/QuantUNetT_rec \\
        --resume      Design_rec_1_si/models/QuantUNetT_rec \\
        --buffer_path Design_rec_1_si/models/replay_buffer.pkl \\
        --buffer_size 300 \\
        --si_lambda   1.0 \\
        --si_xi       1e-4 \\
        --epochs      10 \\
        --batch_size  2 \\
        --gpu         0 \\
        --lr          0.0002 \\
        --step_size   2 \\
        --seed        123

Sanity checks after round 1
----------------------------
  1. si_state.pkl must exist in {train_data}/models/
  2. Log line "Omega mean=..." must be non-zero (importance has accumulated)
  3. Effective penalty scale grows each round as Omega accumulates
"""

import argparse
import os
import random
import re
import time
from enum import Enum

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from QuantUNetT_model import QuantUNetT as PImodel
from replay_buffer import ReplayBuffer

# Only regularise the encoder — decoder is free to adapt each round
ENCODER_PREFIXES = (
    'ConvBlock1.', 'pool1.',
    'ConvBlock2.', 'pool2.',
    'ConvBlock3.', 'pool3.',
    'ConvBlock4.', 'pool4.',
    'ConvBlock5.',
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='SI Trainer (replay + Synaptic Intelligence)')

parser.add_argument('--train_data', default='', metavar='DIR')
parser.add_argument('--pth_name', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--si_state_path', default='', type=str,
                    help='Path to si_state.pkl. Defaults to '
                         '{train_data}/models/si_state.pkl.')

parser.add_argument('--buffer_path', default='', type=str)
parser.add_argument('--buffer_size', default=300, type=int)
parser.add_argument('--si_lambda', default=1.0, type=float,
                    help='SI regularisation strength (c in the SI paper). '
                         'Scales the Omega-weighted penalty.')
parser.add_argument('--si_xi', default=1e-4, type=float,
                    help='Damping constant to prevent division by zero when '
                         'computing Omega update. Default 1e-4.')

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--step_size', default=2, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['adam', 'sgd'])
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--print_freq', default=1, type=int)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class PinballLoss:
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        loss[error < 0] = self.quantile * error[error < 0].abs()
        loss[error > 0] = (1 - self.quantile) * error[error > 0].abs()
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class QuantileLoss(nn.Module):
    def __init__(self, q_lo=0.05, q_hi=0.95,
                 q_lo_weight=1.0, q_hi_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.q_lo_loss = PinballLoss(quantile=q_lo)
        self.q_hi_loss = PinballLoss(quantile=q_hi)
        self.mse_loss = nn.MSELoss()
        self.q_lo_weight = q_lo_weight
        self.q_hi_weight = q_hi_weight
        self.mse_weight = mse_weight

    def forward(self, low, mu, high, target):
        return (self.q_lo_weight * self.q_lo_loss(low, target) +
                self.q_hi_weight * self.q_hi_loss(high, target) +
                self.mse_weight * self.mse_loss(mu, target))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _extract_indices(filename):
    nums = re.findall(r'\d+', os.path.basename(filename))
    return int(nums[0]), int(nums[1])


class ReplayDataset(Dataset):
    def __init__(self, I_paths: list, Phi_paths: list):
        assert len(I_paths) == len(Phi_paths)
        self.I_paths = I_paths
        self.Phi_paths = Phi_paths

    def __len__(self):
        return len(self.I_paths)

    def __getitem__(self, index):
        I = torch.tensor(
            np.load(self.I_paths[index]).astype(np.float32)).unsqueeze(0)
        Phi = torch.tensor(
            np.load(self.Phi_paths[index]).astype(np.float32)).unsqueeze(0)
        return I, Phi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_all_sorted_paths(train_data: str):
    I_dir = os.path.join(train_data, 'training_set', 'intensity', 'npy')
    Phi_dir = os.path.join(train_data, 'training_set', 'phase', 'npy')
    I_files = sorted(os.listdir(I_dir), key=_extract_indices)
    Phi_files = sorted(os.listdir(Phi_dir), key=_extract_indices)
    I_paths = [os.path.join(I_dir, f) for f in I_files]
    Phi_paths = [os.path.join(Phi_dir, f) for f in Phi_files]
    return I_paths, Phi_paths


def save_checkpoint(state: dict, name: str):
    torch.save(state, name + '.pth.tar')


# ---------------------------------------------------------------------------
# Progress utilities
# ---------------------------------------------------------------------------
class Summary(Enum):
    NONE = 0
    AVERAGE = 1


class AverageMeter:
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return ('{name} {val' + self.fmt + '} ({avg' + self.fmt + '})').format(
            **self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=''):
        nd = len(str(num_batches))
        self.fmt = '[{:' + str(nd) + 'd}/' + str(num_batches) + ']'
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.fmt.format(batch)] + [str(m) for m in self.meters]
        print('\t'.join(entries))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(loader, model, criterion, optimizer, epoch, device,
                print_freq, theta_prev, Omega, running_sum, si_lambda):
    """One training epoch with SI penalty and online omega accumulation.

    running_sum is updated in-place: running_sum_i -= grad_i * (theta_i - theta_prev_i)
    This accumulates the path-integral of gradient × displacement across all
    batches and epochs in the round, building up per-parameter importance.
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(loader), [losses], prefix=f'Epoch [{epoch}]')
    model.train()

    for i, (I, Phi) in enumerate(loader):
        I = I.to(device, non_blocking=True)
        Phi = Phi.to(device, non_blocking=True)

        low, mu, high = model(I)
        task_loss = criterion(low, mu, high, Phi)

        # SI penalty: Omega-weighted distance from previous round's encoder
        si_penalty = sum(
            (Omega[n] * (p - theta_prev[n]).pow(2)).sum()
            for n, p in model.named_parameters()
            if n in Omega
        )
        loss = task_loss + (si_lambda / 2.0) * si_penalty

        optimizer.zero_grad()
        loss.backward()

        # Accumulate omega path integral AFTER backward (grads available)
        # before optimizer step (params not yet updated)
        # running_sum_i -= grad_i * (theta_i_current - theta_prev_i)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in running_sum and p.grad is not None:
                    running_sum[n] -= p.grad.data * (p.data - theta_prev[n])

        optimizer.step()

        losses.update(loss.item(), Phi.size(0))
        if i % print_freq == 0:
            progress.display(i + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f'[SITrainer] Device: {device}')

    # ---- Replay buffer ----
    buffer_path = args.buffer_path or os.path.join(
        args.train_data, 'models', 'replay_buffer.pkl')

    if os.path.isfile(buffer_path):
        buffer = ReplayBuffer.load(buffer_path)
        if buffer.max_size != args.buffer_size:
            print(f'[WARNING] Buffer on disk has max_size={buffer.max_size} '
                  f'but --buffer_size={args.buffer_size}. Using disk value.')
    else:
        buffer = ReplayBuffer(max_size=args.buffer_size)
        print(f'[SITrainer] New replay buffer | max_size={args.buffer_size}')

    # ---- Detect new samples ----
    all_I_paths, all_Phi_paths = get_all_sorted_paths(args.train_data)
    n_current = len(all_I_paths)
    n_prev = buffer.n_trained

    new_I_paths = all_I_paths[n_prev:]
    new_Phi_paths = all_Phi_paths[n_prev:]

    print(f'[SITrainer] Dataset: {n_current} total | '
          f'{n_prev} seen before | {len(new_I_paths)} new this round')

    # ---- Build combined training set ----
    buf_I, buf_Phi = buffer.get_paths()
    combined_I = new_I_paths + buf_I
    combined_Phi = new_Phi_paths + buf_Phi

    if not combined_I:
        print('[SITrainer] No samples available. Exiting.')
        return

    print(f'[SITrainer] Training on {len(new_I_paths)} new + '
          f'{len(buf_I)} buffer = {len(combined_I)} samples | '
          f'si_lambda={args.si_lambda}')

    dataset = ReplayDataset(combined_I, combined_Phi)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = PImodel().to(device)

    # ---- Warm start ----
    resume_path = (args.resume + '.pth.tar') if args.resume else ''
    if resume_path and os.path.isfile(resume_path):
        loc = str(device)
        ckpt = torch.load(resume_path, map_location=loc)
        model.load_state_dict(ckpt['state_dict'])
        print(f'[SITrainer] Warm-start from epoch {ckpt["epoch"]} '
              f'← {resume_path}')
    elif args.resume:
        print(f'[WARNING] Checkpoint not found at {resume_path}. '
              f'Training from random init.')
    else:
        print('[SITrainer] No --resume given. Cold start (first round).')

    # ---- SI state: load or initialise ----
    si_state_path = args.si_state_path or os.path.join(
        args.train_data, 'models', 'si_state.pkl')

    if os.path.isfile(si_state_path):
        state = torch.load(si_state_path, map_location=str(device))
        Omega = {n: t.to(device) for n, t in state['Omega'].items()}
        theta_prev = {n: t.to(device) for n, t in state['theta_prev'].items()}
        omega_mean = float(torch.stack([o.mean() for o in Omega.values()]).mean())
        print(f'[SITrainer] Loaded SI state from {si_state_path} | '
              f'{len(Omega)} encoder tensors | Omega mean={omega_mean:.6e}')
    else:
        # Round 1: Omega=0, anchor = current (cold-start) encoder weights
        Omega = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if n.startswith(ENCODER_PREFIXES)
        }
        theta_prev = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if n.startswith(ENCODER_PREFIXES)
        }
        print(f'[SITrainer] No SI state found → initialising | '
              f'{len(Omega)} encoder tensors | Omega=0 (round 1 = pure replay)')

    # running_sum accumulates gradient×displacement across all batches this round
    running_sum = {n: torch.zeros_like(p) for n, p in Omega.items()}

    # ---- Optimizer & scheduler ----
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = QuantileLoss().to(device)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # ---- Train ----
    for epoch in range(args.epochs):
        train_epoch(loader, model, criterion, optimizer, epoch, device,
                    args.print_freq, theta_prev, Omega, running_sum, args.si_lambda)
        scheduler.step()

    # ---- Update Omega with this round's contributions ----
    # delta_i = how much each encoder param moved this round
    # Omega_i += clamp(running_sum_i, min=0) / (delta_i^2 + xi)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in Omega:
                delta = p.detach() - theta_prev[n]
                Omega[n] += (running_sum[n] / (delta.pow(2) + args.si_xi)).clamp(min=0)

    omega_mean_new = float(torch.stack([o.mean() for o in Omega.values()]).mean())
    print(f'[SITrainer] Omega updated | mean={omega_mean_new:.6e} | '
          f'effective penalty scale ≈ si_lambda/2 * Omega_mean = '
          f'{args.si_lambda / 2 * omega_mean_new:.6e}')

    # ---- Save checkpoint ----
    pth_name = args.pth_name or os.path.join(
        args.train_data, 'models',
        f'QuantUNetT_{os.path.basename(args.train_data)}')
    save_checkpoint(
        {
            'epoch': args.epochs,
            'state_dict': model.state_dict(),
            'best_acc1': float('inf'),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
        pth_name,
    )
    print(f'[SITrainer] Checkpoint saved → {pth_name}.pth.tar')

    # ---- Save SI state for next round ----
    # theta_prev for next round = current encoder weights (end of this round)
    theta_prev_new = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith(ENCODER_PREFIXES)
    }
    torch.save({'Omega': Omega, 'theta_prev': theta_prev_new}, si_state_path)
    print(f'[SITrainer] SI state saved → {si_state_path}')

    # ---- Update replay buffer ----
    buffer.update(new_I_paths, new_Phi_paths)
    buffer.n_trained = n_current
    buffer.save(buffer_path)

    elapsed = time.perf_counter() - t_start
    print(f'[SITrainer] Done in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
