"""ddpg.py -- Deep Deterministic Policy Gradient (Lillicrap et al., 2016).

Deterministic actor mu(s) with additive Gaussian exploration noise; critic
Q(s,a) regressed on the stochastic reward. Because episodes are single-step,
the critic target is the immediate reward (no bootstrapping / no discount),
so the critic learns the EXPECTED ADR surface E[r | s, a] under the random
UE orientation -- exactly the objective of Problem (P1) in the stochastic
setting of the paper.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import mlp, Replay, set_seeds

HID = 64


def train(env, total_steps=6000, seed=0, lr_a=3e-4, lr_c=1e-3,
          noise=0.30, batch=128, warmup=200, label="DDPG"):
    set_seeds(seed)
    sdim, adim = env.state_dim, env.action_dim

    actor = mlp([sdim, HID, HID, adim], nn.Tanh())
    critic = mlp([sdim + adim, HID, HID, 1])
    opt_a = torch.optim.Adam(actor.parameters(), lr_a)
    opt_c = torch.optim.Adam(critic.parameters(), lr_c)
    buf = Replay(sdim, adim)

    s = env.reset()
    curve = []
    for t in range(total_steps):
        st = torch.as_tensor(s).unsqueeze(0)
        with torch.no_grad():
            a = actor(st).squeeze(0).numpy()
        if t < warmup:                                    # uniform warm start
            a = np.random.uniform(-1, 1, adim)
        else:
            a = np.clip(a + noise * np.random.randn(adim), -1, 1)
        _, r, _, info = env.step(a)
        buf.add(s, a, r)
        curve.append(info["adr_mean"])

        if buf.n >= batch:
            bs_, ba, br = buf.sample(batch)
            # critic: regress expected (stochastic) reward
            loss_c = F.mse_loss(critic(torch.cat([bs_, ba], 1)), br)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()
            # actor: deterministic policy gradient through the critic
            loss_a = -critic(torch.cat([bs_, actor(bs_)], 1)).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()

    with torch.no_grad():
        a_fin = actor(torch.as_tensor(s).unsqueeze(0)).squeeze(0).numpy()
    return label, np.asarray(curve), a_fin
