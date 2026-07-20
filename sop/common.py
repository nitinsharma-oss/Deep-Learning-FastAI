"""common.py -- shared building blocks for all DRL agents (kept minimal so
each algorithm file remains readable on its own)."""
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"


def mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Replay:
    """Replay buffer for single-step (bandit) episodes: stores (s, a, r)."""

    def __init__(self, sdim, adim, cap=40000):
        self.s = np.zeros((cap, sdim), np.float32)
        self.a = np.zeros((cap, adim), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.n, self.cap = 0, cap

    def add(self, s, a, r):
        i = self.n % self.cap
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.n += 1

    def sample(self, bs):
        idx = np.random.randint(0, min(self.n, self.cap), bs)
        t = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return t(self.s), t(self.a), t(self.r)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
