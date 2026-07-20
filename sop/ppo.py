"""ppo.py -- Proximal Policy Optimization (Schulman et al., 2017),
clipped-surrogate variant with a learned value baseline.

Improvements over the earlier under-performing version:
  * larger on-policy batch (256) so the advantage estimate under the NOISY
    stochastic ADR reward is meaningful;
  * more optimisation epochs per batch (15) with early stopping on an
    approximate-KL trigger;
  * decaying entropy bonus: strong exploration early, sharp policy late;
  * state-independent learned log-std initialised wide (exp(-0.1) ~ 0.9).
Single-step episodes: return = reward, advantage = r - V(s).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import mlp, set_seeds

HID = 64


def train(env, total_steps=6000, seed=0, lr=3e-4, batch=256, epochs=15,
          clip=0.2, kl_stop=0.03, label="PPO"):
    set_seeds(seed)
    sdim, adim = env.state_dim, env.action_dim

    trunk = mlp([sdim, HID, HID])
    mu_l = nn.Linear(HID, adim)
    log_std = nn.Parameter(torch.zeros(adim) - 0.1)
    vf = mlp([sdim, HID, HID, 1])
    opt = torch.optim.Adam(list(trunk.parameters()) + list(mu_l.parameters())
                           + [log_std] + list(vf.parameters()), lr)

    def dist(st):
        mu = torch.tanh(mu_l(F.relu(trunk(st))))
        return torch.distributions.Normal(mu, log_std.exp())

    s = env.reset()
    curve, steps, it = [], 0, 0
    n_iters = max(1, total_steps // batch)
    while steps < total_steps:
        S, A, R, LP = [], [], [], []
        for _ in range(batch):
            st = torch.as_tensor(s).unsqueeze(0)
            with torch.no_grad():
                d = dist(st)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
            _, r, _, info = env.step(np.clip(a.squeeze(0).numpy(), -1, 1))
            S.append(s); A.append(a.squeeze(0).numpy()); R.append(r)
            LP.append(lp.item()); curve.append(info["adr_mean"]); steps += 1

        S = torch.as_tensor(np.asarray(S)); A = torch.as_tensor(np.asarray(A))
        R = torch.as_tensor(np.asarray(R), dtype=torch.float32).unsqueeze(1)
        LP = torch.as_tensor(np.asarray(LP), dtype=torch.float32).unsqueeze(1)
        ent_coef = 0.01 * max(0.0, 1.0 - it / (0.6 * n_iters))   # decay to 0

        for _ in range(epochs):
            d = dist(S)
            lp = d.log_prob(A).sum(-1, keepdim=True)
            if (LP - lp).mean().item() > kl_stop:                # early stop
                break
            v = vf(S)
            adv = (R - v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            ratio = torch.exp(lp - LP)
            l_pi = -torch.min(ratio * adv,
                              torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            loss = l_pi + 0.5 * F.mse_loss(v, R) \
                   - ent_coef * d.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        it += 1

    with torch.no_grad():
        a_fin = torch.tanh(mu_l(F.relu(trunk(torch.as_tensor(s).unsqueeze(0)))))
    return label, np.asarray(curve[:total_steps]), a_fin.squeeze(0).numpy()
