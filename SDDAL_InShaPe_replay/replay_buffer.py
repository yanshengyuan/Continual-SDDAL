"""
replay_buffer.py — Fixed-size experience replay buffer using reservoir sampling.

Maintains a representative subset of all training samples seen across SDDAL
rounds without unbounded memory growth. Persists to disk between rounds.
"""

import os
import pickle
import random


class ReplayBuffer:
    """
    Reservoir-sampling replay buffer for continual learning in SDDAL.

    Reservoir sampling guarantees every historical sample has an equal
    probability of being in the buffer, so the buffer stays representative
    of the full training distribution even as the dataset grows.

    Attributes
    ----------
    max_size  : hard cap on buffer entries
    n_seen    : total samples ever offered (used for reservoir math)
    n_trained : total dataset size at the last ContinualTrainer call
                (used to identify newly-added samples on the next call)
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.I_paths: list = []    # intensity .npy paths
        self.Phi_paths: list = []  # phase .npy paths
        self.n_seen: int = 0
        self.n_trained: int = 0

    # ------------------------------------------------------------------
    # Core reservoir update
    # ------------------------------------------------------------------
    def update(self, new_I_paths: list, new_Phi_paths: list):
        """
        Incorporate new samples into the buffer via reservoir sampling.

        Call this AFTER training, passing only the paths of samples added
        since the previous training round.
        """
        for i_path, phi_path in zip(new_I_paths, new_Phi_paths):
            self.n_seen += 1
            if len(self.I_paths) < self.max_size:
                # Buffer not yet full — always accept
                self.I_paths.append(i_path)
                self.Phi_paths.append(phi_path)
            else:
                # Replace a random existing entry with probability max_size / n_seen
                j = random.randint(0, self.n_seen - 1)
                if j < self.max_size:
                    self.I_paths[j] = i_path
                    self.Phi_paths[j] = phi_path

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_paths(self):
        """Return (I_paths, Phi_paths) copies of current buffer contents."""
        return list(self.I_paths), list(self.Phi_paths)

    def __len__(self):
        return len(self.I_paths)

    def __repr__(self):
        return (f'ReplayBuffer(size={len(self)}/{self.max_size}, '
                f'n_seen={self.n_seen}, n_trained={self.n_trained})')

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Serialize buffer state to disk."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        state = {
            'max_size': self.max_size,
            'I_paths': self.I_paths,
            'Phi_paths': self.Phi_paths,
            'n_seen': self.n_seen,
            'n_trained': self.n_trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'[ReplayBuffer] Saved  {len(self):>5}/{self.max_size} samples '
              f'| n_seen={self.n_seen} n_trained={self.n_trained} → {path}')

    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        """Deserialize buffer state from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        buf = cls(max_size=state['max_size'])
        buf.I_paths = state['I_paths']
        buf.Phi_paths = state['Phi_paths']
        buf.n_seen = state['n_seen']
        buf.n_trained = state.get('n_trained', len(buf.I_paths))
        print(f'[ReplayBuffer] Loaded {len(buf):>5}/{buf.max_size} samples '
              f'| n_seen={buf.n_seen} n_trained={buf.n_trained} ← {path}')
        return buf
