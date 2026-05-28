"""
ContinualTrainer.py — Drop-in replacement for Trainer.py using experience replay.

Instead of retraining from scratch on the entire (ever-growing) dataset every
AL round, this script:

  1. Warm-starts from the previous checkpoint (no weight reset).
  2. Detects only the samples added since the last training call.
  3. Combines those new samples with a fixed-size replay buffer.
  4. Trains on that fixed-size combined set (constant cost per round).
  5. Updates the replay buffer via reservoir sampling and saves it.

Training set size per round: (n_new_this_round + buffer_size)
This stays constant regardless of total AL iterations, unlike full retraining
whose cost grows with every new sample.

Usage
-----
    python3 ContinualTrainer.py \
        --train_data  Design_rec \
        --pth_name    Design_rec/models/QuantUNetT_rec \
        --resume      Design_rec/models/QuantUNetT_rec \
        --buffer_path Design_rec/models/replay_buffer.pkl \
        --buffer_size 500 \
        --epochs      10 \
        --batch_size  2 \
        --gpu         0 \
        --lr          0.0002 \
        --step_size   2 \
        --seed        123
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Continual Trainer with Experience Replay')

# Data / checkpoint
parser.add_argument('--train_data', default='', metavar='DIR',
                    help='Design directory (e.g. Design_rec)')
parser.add_argument('--pth_name', default='', type=str,
                    help='Output checkpoint path without .pth.tar extension')
parser.add_argument('--resume', default='', type=str,
                    help='Previous checkpoint path without .pth.tar (warm start). '
                         'Leave empty for cold start on first round.')

# Replay buffer
parser.add_argument('--buffer_path', default='', type=str,
                    help='Path to replay buffer .pkl file. '
                         'Defaults to <train_data>/models/replay_buffer.pkl.')
parser.add_argument('--buffer_size', default=500, type=int,
                    help='Maximum number of samples kept in the replay buffer.')

# Training hyperparameters
parser.add_argument('--epochs', default=10, type=int,
                    help='Epochs per continual update. Fewer are needed vs. '
                         'full retraining because of warm starting (default: 10).')
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
# Loss (identical to Trainer.py)
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
# Dataset that loads from explicit path lists (new + buffer samples)
# ---------------------------------------------------------------------------
def _extract_indices(filename):
    nums = re.findall(r'\d+', os.path.basename(filename))
    return int(nums[0]), int(nums[1])


class ReplayDataset(Dataset):
    """Loads (intensity, phase) pairs from explicit .npy file path lists."""

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
    """Return all (I_paths, Phi_paths) sorted by file index from training_set."""
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
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_epoch(loader, model, criterion, optimizer, epoch, device, print_freq):
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(loader), [losses], prefix=f'Epoch [{epoch}]')
    model.train()
    for i, (I, Phi) in enumerate(loader):
        I = I.to(device, non_blocking=True)
        Phi = Phi.to(device, non_blocking=True)
        low, mu, high = model(I)
        loss = criterion(low, mu, high, Phi)
        losses.update(loss.item(), Phi.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

    # ---- Device ----
    if torch.cuda.is_available() and args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f'[ContinualTrainer] Device: {device}')

    # ---- Replay buffer ----
    buffer_path = args.buffer_path or os.path.join(
        args.train_data, 'models', 'replay_buffer.pkl')

    if os.path.isfile(buffer_path):
        buffer = ReplayBuffer.load(buffer_path)
        # Warn if caller changed --buffer_size after buffer was created
        if buffer.max_size != args.buffer_size:
            print(f'[WARNING] Buffer on disk has max_size={buffer.max_size} '
                  f'but --buffer_size={args.buffer_size}. Using disk value.')
    else:
        buffer = ReplayBuffer(max_size=args.buffer_size)
        print(f'[ContinualTrainer] New replay buffer | max_size={args.buffer_size}')

    # ---- Detect new samples ----
    # buffer.n_trained records total dataset size after the last training call.
    # All files beyond that index are new.
    all_I_paths, all_Phi_paths = get_all_sorted_paths(args.train_data)
    n_current = len(all_I_paths)
    n_prev = buffer.n_trained

    new_I_paths = all_I_paths[n_prev:]
    new_Phi_paths = all_Phi_paths[n_prev:]

    print(f'[ContinualTrainer] Dataset: {n_current} total | '
          f'{n_prev} seen before | {len(new_I_paths)} new this round')

    # ---- Build combined training set ----
    buf_I, buf_Phi = buffer.get_paths()
    combined_I = new_I_paths + buf_I
    combined_Phi = new_Phi_paths + buf_Phi

    if not combined_I:
        print('[ContinualTrainer] No samples available. Exiting.')
        return

    print(f'[ContinualTrainer] Training on {len(new_I_paths)} new + '
          f'{len(buf_I)} buffer = {len(combined_I)} samples '
          f'(vs {n_current} for full retraining)')

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

    # ---- Warm start from previous checkpoint ----
    resume_path = (args.resume + '.pth.tar') if args.resume else ''
    if resume_path and os.path.isfile(resume_path):
        loc = str(device)
        ckpt = torch.load(resume_path, map_location=loc)
        model.load_state_dict(ckpt['state_dict'])
        print(f'[ContinualTrainer] Warm-start from epoch {ckpt["epoch"]} '
              f'← {resume_path}')
    elif args.resume:
        print(f'[WARNING] Checkpoint not found at {resume_path}. '
              f'Training from random init.')
    else:
        print('[ContinualTrainer] No --resume given. Cold start (first round).')

    # ---- Optimizer & scheduler (always fresh — consistent with Trainer.py) ----
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
                    args.print_freq)
        scheduler.step()

    # ---- Save checkpoint (same path as Trainer.py so Scanner.py can find it) ----
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
    print(f'[ContinualTrainer] Checkpoint saved → {pth_name}.pth.tar')

    # ---- Update buffer with new samples ----
    buffer.update(new_I_paths, new_Phi_paths)
    buffer.n_trained = n_current
    buffer.save(buffer_path)

    elapsed = time.perf_counter() - t_start
    print(f'[ContinualTrainer] Done in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
