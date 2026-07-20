"""capg.py -- Continuous Action Policy Gradient.

Vanilla likelihood-ratio (REINFORCE-style) policy gradient with a Gaussian
policy over the continuous action a = [gamma, omega], improved with:
  * an exponential moving-average reward baseline (variance reduction),
  * a small entropy bonus with linear decay (sustained early exploration),
  * a learned but floored log-std so the policy cannot collapse prematurely
    while the stochastic ADR reward is still noisy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import mlp, set_seeds

HID = 64
LOG_STD_FLOOR = -3.0


def train(env, total_steps=6000, seed=0, lr=3e-4, label="CAPG"):
    set_seeds(seed)
    sdim, adim = env.state_dim, env.action_dim

    trunk = mlp([sdim, HID, HID])
    mu_l = nn.Linear(HID, adim)
    log_std = nn.Parameter(torch.zeros(adim) - 0.1)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(mu_l.parameters())
                           + [log_std], lr)

    s = env.reset()
    baseline, curve = 0.0, []
    for t in range(total_steps):
        st = torch.as_tensor(s).unsqueeze(0)
        mu = torch.tanh(mu_l(F.relu(trunk(st))))
        std = torch.clamp(log_std, min=LOG_STD_FLOOR).exp()
        d = torch.distributions.Normal(mu, std)
        a = d.sample()
        _, r, _, info = env.step(np.clip(a.squeeze(0).numpy(), -1, 1))
        curve.append(info["adr_mean"])

        baseline = 0.99 * baseline + 0.01 * r                 # EMA baseline
        ent_coef = 0.005 * max(0.0, 1.0 - t / (0.6 * total_steps))
        loss = -(d.log_prob(a).sum()) * (r - baseline) \
               - ent_coef * d.entropy().sum()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        a_fin = torch.tanh(mu_l(F.relu(trunk(torch.as_tensor(s).unsqueeze(0)))))
    return label, np.asarray(curve), a_fin.squeeze(0).numpy()
