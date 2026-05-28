"""
DERTrainer.py — Drop-in replacement for ContinualTrainer.py using DER.

Dark Experience Replay (Buzzega et al., 2020) extends experience replay by
also storing the model's mu prediction at the time each sample enters the
buffer. On future rounds, a distillation loss penalises the current model
for drifting away from those stored predictions:

    L = QuantileLoss(low, mu, high, target)           [all samples]
      + alpha * MSE(mu_current, mu_stored)             [buffer samples only]

This discourages catastrophic forgetting on already-seen data without
requiring a separate frozen teacher model.

Prediction storage: after each round's training, the trained model runs
inference on the new samples and stores mu (shape 1×H×W). So stored
predictions reflect the model's best understanding of each sample at
insertion time.

Usage
-----
    python3 DERTrainer.py \
        --train_data  Design_rec_1_der \
        --pth_name    Design_rec_1_der/models/QuantUNetT_rec \
        --resume      Design_rec_1_der/models/QuantUNetT_rec \
        --buffer_path Design_rec_1_der/models/der_buffer.pkl \
        --buffer_size 500 \
        --alpha       0.5 \
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
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from QuantUNetT_model import QuantUNetT as PImodel
from der_buffer import DERBuffer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DER Trainer (replay + distillation)')

parser.add_argument('--train_data', default='', metavar='DIR')
parser.add_argument('--pth_name', default='', type=str)
parser.add_argument('--resume', default='', type=str)

parser.add_argument('--buffer_path', default='', type=str)
parser.add_argument('--buffer_size', default=500, type=int)
parser.add_argument('--alpha', default=0.5, type=float,
                    help='Weight of the distillation loss on buffer samples.')

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
# Loss (identical to ContinualTrainer.py)
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


class DERDataset(Dataset):
    """
    Loads (I, Phi) pairs and optionally a stored mu prediction.

    mu_refs is a list parallel to I_paths/Phi_paths:
      - None  → new sample, no stored prediction (has_ref = False)
      - ndarray (1, H, W) → buffer sample with stored prediction (has_ref = True)
    """

    def __init__(self, I_paths: list, Phi_paths: list, mu_refs: list):
        assert len(I_paths) == len(Phi_paths) == len(mu_refs)
        self.I_paths = I_paths
        self.Phi_paths = Phi_paths
        self.mu_refs = mu_refs

    def __len__(self):
        return len(self.I_paths)

    def __getitem__(self, idx):
        I = torch.tensor(
            np.load(self.I_paths[idx]).astype(np.float32)).unsqueeze(0)
        Phi = torch.tensor(
            np.load(self.Phi_paths[idx]).astype(np.float32)).unsqueeze(0)
        if self.mu_refs[idx] is not None:
            mu_ref = torch.tensor(self.mu_refs[idx].astype(np.float32))
            has_ref = torch.tensor(True)
        else:
            mu_ref = torch.zeros_like(Phi)
            has_ref = torch.tensor(False)
        return I, Phi, mu_ref, has_ref


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
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_epoch(loader, model, criterion, optimizer, epoch, device,
                print_freq, alpha):
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(loader), [losses], prefix=f'Epoch [{epoch}]')
    model.train()
    for i, (I, Phi, mu_ref, has_ref) in enumerate(loader):
        I = I.to(device, non_blocking=True)
        Phi = Phi.to(device, non_blocking=True)
        mu_ref = mu_ref.to(device, non_blocking=True)
        has_ref = has_ref.to(device, non_blocking=True)

        low, mu, high = model(I)
        loss = criterion(low, mu, high, Phi)

        if alpha > 0.0 and has_ref.any():
            dist_loss = F.mse_loss(mu[has_ref], mu_ref[has_ref])
            loss = loss + alpha * dist_loss

        losses.update(loss.item(), Phi.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_freq == 0:
            progress.display(i + 1)


# ---------------------------------------------------------------------------
# Inference: get mu predictions for a list of intensity paths
# ---------------------------------------------------------------------------
def predict_mu(model, I_paths: list, device) -> list:
    """Run model in eval mode; return list of np.ndarray (1, H, W) per sample."""
    model.eval()
    mu_preds = []
    with torch.no_grad():
        for i_path in I_paths:
            I = torch.tensor(
                np.load(i_path).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, H, W)
            _, mu, _ = model(I)
            mu_preds.append(mu.squeeze(0).cpu().numpy())  # (1, H, W)
    return mu_preds


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
    print(f'[DERTrainer] Device: {device}')

    # ---- DER buffer ----
    buffer_path = args.buffer_path or os.path.join(
        args.train_data, 'models', 'der_buffer.pkl')

    if os.path.isfile(buffer_path):
        buffer = DERBuffer.load(buffer_path)
        if buffer.max_size != args.buffer_size:
            print(f'[WARNING] Buffer on disk has max_size={buffer.max_size} '
                  f'but --buffer_size={args.buffer_size}. Using disk value.')
    else:
        buffer = DERBuffer(max_size=args.buffer_size)
        print(f'[DERTrainer] New DER buffer | max_size={args.buffer_size}')

    # ---- Detect new samples ----
    all_I_paths, all_Phi_paths = get_all_sorted_paths(args.train_data)
    n_current = len(all_I_paths)
    n_prev = buffer.n_trained

    new_I_paths = all_I_paths[n_prev:]
    new_Phi_paths = all_Phi_paths[n_prev:]

    print(f'[DERTrainer] Dataset: {n_current} total | '
          f'{n_prev} seen before | {len(new_I_paths)} new this round')

    # ---- Build combined training set ----
    buf_I, buf_Phi, buf_mu = buffer.get_all()

    # New samples have no stored prediction (None); buffer samples do.
    combined_I = new_I_paths + buf_I
    combined_Phi = new_Phi_paths + buf_Phi
    combined_mu_refs = [None] * len(new_I_paths) + buf_mu

    if not combined_I:
        print('[DERTrainer] No samples available. Exiting.')
        return

    print(f'[DERTrainer] Training on {len(new_I_paths)} new + '
          f'{len(buf_I)} buffer = {len(combined_I)} samples '
          f'(vs {n_current} for full retraining) | alpha={args.alpha}')

    dataset = DERDataset(combined_I, combined_Phi, combined_mu_refs)
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
        print(f'[DERTrainer] Warm-start from epoch {ckpt["epoch"]} '
              f'← {resume_path}')
    elif args.resume:
        print(f'[WARNING] Checkpoint not found at {resume_path}. '
              f'Training from random init.')
    else:
        print('[DERTrainer] No --resume given. Cold start (first round).')

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
                    args.print_freq, args.alpha)
        scheduler.step()

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
    print(f'[DERTrainer] Checkpoint saved → {pth_name}.pth.tar')

    # ---- Get mu predictions for new samples (with trained model) ----
    if new_I_paths:
        print(f'[DERTrainer] Running inference on {len(new_I_paths)} new samples...')
        new_mu_preds = predict_mu(model, new_I_paths, device)
    else:
        new_mu_preds = []

    # ---- Update DER buffer ----
    buffer.update(new_I_paths, new_Phi_paths, new_mu_preds)
    buffer.n_trained = n_current
    buffer.save(buffer_path)

    elapsed = time.perf_counter() - t_start
    print(f'[DERTrainer] Done in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
