"""
der_buffer.py — Fixed-size DER replay buffer storing paths + model predictions.

Same reservoir-sampling logic as replay_buffer.py, but additionally stores
the model's mu prediction (centre quantile, shape 1×H×W) at the time each
sample enters the buffer. DERTrainer.py reads these stored predictions to
compute the distillation loss during future rounds.
"""

import os
import pickle
import random


class DERBuffer:
    """
    Reservoir-sampling replay buffer with stored model predictions (DER).

    Attributes
    ----------
    max_size  : hard cap on buffer entries
    n_seen    : total samples ever offered (used for reservoir math)
    n_trained : total dataset size at the last DERTrainer call
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.I_paths: list = []       # intensity .npy paths
        self.Phi_paths: list = []     # phase .npy paths
        self.mu_preds: list = []      # np.ndarray (1, H, W) per sample
        self.n_seen: int = 0
        self.n_trained: int = 0

    def update(self, new_I_paths: list, new_Phi_paths: list, new_mu_preds: list):
        """
        Incorporate new samples into the buffer via reservoir sampling.

        Call this AFTER training so mu_preds reflect the trained model.
        new_mu_preds must be a list of np.ndarray with shape (1, H, W).
        """
        for i_path, phi_path, mu in zip(new_I_paths, new_Phi_paths, new_mu_preds):
            self.n_seen += 1
            if len(self.I_paths) < self.max_size:
                self.I_paths.append(i_path)
                self.Phi_paths.append(phi_path)
                self.mu_preds.append(mu)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.max_size:
                    self.I_paths[j] = i_path
                    self.Phi_paths[j] = phi_path
                    self.mu_preds[j] = mu

    def get_all(self):
        """Return (I_paths, Phi_paths, mu_preds) copies of current buffer contents."""
        return list(self.I_paths), list(self.Phi_paths), list(self.mu_preds)

    def __len__(self):
        return len(self.I_paths)

    def __repr__(self):
        return (f'DERBuffer(size={len(self)}/{self.max_size}, '
                f'n_seen={self.n_seen}, n_trained={self.n_trained})')

    def save(self, path: str):
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        state = {
            'max_size': self.max_size,
            'I_paths': self.I_paths,
            'Phi_paths': self.Phi_paths,
            'mu_preds': self.mu_preds,
            'n_seen': self.n_seen,
            'n_trained': self.n_trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'[DERBuffer] Saved  {len(self):>5}/{self.max_size} samples '
              f'| n_seen={self.n_seen} n_trained={self.n_trained} → {path}')

    @classmethod
    def load(cls, path: str) -> 'DERBuffer':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        buf = cls(max_size=state['max_size'])
        buf.I_paths = state['I_paths']
        buf.Phi_paths = state['Phi_paths']
        buf.mu_preds = state['mu_preds']
        buf.n_seen = state['n_seen']
        buf.n_trained = state.get('n_trained', len(buf.I_paths))
        print(f'[DERBuffer] Loaded {len(buf):>5}/{buf.max_size} samples '
              f'| n_seen={buf.n_seen} n_trained={buf.n_trained} ← {path}')
        return buf
