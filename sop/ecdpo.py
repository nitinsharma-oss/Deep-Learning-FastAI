"""ecdpo.py -- Enhanced Continuous Deep Policy Optimization.

*** PLACEHOLDER IMPLEMENTATION -- substitute your actual ECDPO update rule ***

Since the true ECDPO algorithm is proposed in the authors' paper and its
update equations were not provided, this file implements a plausible
"enhanced" continuous policy optimizer on a DDPG backbone with four
enhancements that are individually well-established:

  E1. Twin critics with a minimum operator (clipped double-Q, TD3-style):
      counters critic over-estimation caused by the noisy stochastic ADR
      reward from the random UE orientation.
  E2. Delayed policy updates (actor updated every 2 critic updates):
      the actor always ascends a better-fitted value surface.
  E3. Decaying exploration noise (sigma: 0.5 -> 0.05 exponentially):
      broad global search early, fine alignment of (gamma, omega) late.
  E4. Huber (smooth-L1) critic loss: robust to outlier rewards produced by
      rare tail draws of the truncated-Laplace orientation.

Every enhancement is a keyword flag, so ablations are one-liners and the
class can be rewired to the real ECDPO with minimal edits.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import mlp, Replay, set_seeds

HID = 64


def train(env, total_steps=6000, seed=0, lr_a=3e-4, lr_c=1e-3, batch=128,
          warmup=200, noise0=0.5, noise_min=0.05, noise_decay=0.9992,
          twin=True, delayed=True, huber=True, label="ECDPO"):
    set_seeds(seed)
    sdim, adim = env.state_dim, env.action_dim

    actor = mlp([sdim, HID, HID, adim], nn.Tanh())
    q1 = mlp([sdim + adim, HID, HID, 1])
    q2 = mlp([sdim + adim, HID, HID, 1]) if twin else None
    opt_a = torch.optim.Adam(actor.parameters(), lr_a)
    params_c = list(q1.parameters()) + (list(q2.parameters()) if twin else [])
    opt_c = torch.optim.Adam(params_c, lr_c)
    buf = Replay(sdim, adim)
    critic_loss = F.smooth_l1_loss if huber else F.mse_loss

    s = env.reset()
    noise = noise0
    curve = []
    for t in range(total_steps):
        st = torch.as_tensor(s).unsqueeze(0)
        with torch.no_grad():
            a = actor(st).squeeze(0).numpy()
        if t < warmup:
            a = np.random.uniform(-1, 1, adim)
        else:
            a = np.clip(a + noise * np.random.randn(adim), -1, 1)   # E3
        _, r, _, info = env.step(a)
        buf.add(s, a, r)
        curve.append(info["adr_mean"])
        noise = max(noise_min, noise * noise_decay)                  # E3

        if buf.n >= batch:
            bs_, ba, br = buf.sample(batch)
            qin = torch.cat([bs_, ba], 1)
            loss_c = critic_loss(q1(qin), br)                        # E4
            if twin:
                loss_c = loss_c + critic_loss(q2(qin), br)           # E1
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

            if (not delayed) or (t % 2 == 0):                        # E2
                pa = actor(bs_)
                qv = q1(torch.cat([bs_, pa], 1))
                if twin:
                    qv = torch.min(qv, q2(torch.cat([bs_, pa], 1)))  # E1
                loss_a = -qv.mean()
                opt_a.zero_grad(); loss_a.backward(); opt_a.step()

    with torch.no_grad():
        a_fin = actor(torch.as_tensor(s).unsqueeze(0)).squeeze(0).numpy()
    return label, np.asarray(curve), a_fin
