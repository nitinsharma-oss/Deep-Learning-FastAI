"""
Solve Problem (P1) with five DRL algorithms under identical conditions:
same environment, state, action space [-1,1]^2 -> (gamma, omega), and reward (ADR).

DDPG  : deterministic actor-critic, replay buffer, Gaussian exploration noise.
SAC   : stochastic tanh-Gaussian actor, twin soft Q-critics, fixed temperature.
PPO   : clipped-surrogate on-policy Gaussian policy with value baseline.
CAPG  : continuous-action policy gradient (REINFORCE, Gaussian policy,
        moving-average baseline).
ECDPO : "enhanced continuous deep policy optimization" -- implemented here as a
        DDPG backbone enhanced with twin critics, delayed policy updates and
        decaying exploration noise. NOTE: placeholder -- substitute the actual
        ECDPO update rule from the paper.

The task is a single-step (contextual-bandit) episode, matching the problem
formulation: the agent outputs a_t = [gamma_t, omega_t], receives ADR as
reward, episode ends.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oirs_env import OIRSEnv, analytic_optimum, adr

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"
TOTAL_STEPS = 4000
HID = 64


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
    def __init__(self, sdim, adim, cap=20000):
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


# --------------------------------- DDPG -------------------------------------
def run_ddpg(env, twin=False, delayed=False, noise0=0.4, noise_decay=1.0,
             label="DDPG"):
    """DDPG; with twin critics + delayed updates + noise decay it becomes the
    ECDPO placeholder."""
    sdim, adim = env.state_dim, env.action_dim
    actor = mlp([sdim, HID, HID, adim], nn.Tanh())
    q1 = mlp([sdim + adim, HID, HID, 1])
    q2 = mlp([sdim + adim, HID, HID, 1]) if twin else None
    opt_a = torch.optim.Adam(actor.parameters(), 3e-4)
    params_c = list(q1.parameters()) + (list(q2.parameters()) if twin else [])
    opt_c = torch.optim.Adam(params_c, 1e-3)
    buf = Replay(sdim, adim)

    s = env.reset()
    noise = noise0
    curve = []
    for t in range(TOTAL_STEPS):
        st = torch.as_tensor(s).unsqueeze(0)
        with torch.no_grad():
            a = actor(st).squeeze(0).numpy()
        a = np.clip(a + noise * np.random.randn(adim), -1, 1)
        _, r, _, info = env.step(a)
        buf.add(s, a, r)
        curve.append(info["adr"])
        noise *= noise_decay

        if buf.n >= 128:
            bs_, ba, br = buf.sample(128)
            # critic(s): single-step episode -> target is the reward itself
            qin = torch.cat([bs_, ba], 1)
            loss_c = F.mse_loss(q1(qin), br)
            if twin:
                loss_c = loss_c + F.mse_loss(q2(qin), br)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

            if (not delayed) or (t % 2 == 0):
                pa = actor(bs_)
                qv = q1(torch.cat([bs_, pa], 1))
                if twin:
                    qv = torch.min(qv, q2(torch.cat([bs_, pa], 1)))
                loss_a = -qv.mean()
                opt_a.zero_grad(); loss_a.backward(); opt_a.step()

    with torch.no_grad():
        a_fin = actor(torch.as_tensor(s).unsqueeze(0)).squeeze(0).numpy()
    return label, np.array(curve), a_fin


# ---------------------------------- SAC -------------------------------------
def run_sac(env, alpha=0.003):
    sdim, adim = env.state_dim, env.action_dim
    trunk = mlp([sdim, HID, HID])
    mu_l, ls_l = nn.Linear(HID, adim), nn.Linear(HID, adim)
    q1 = mlp([sdim + adim, HID, HID, 1])
    q2 = mlp([sdim + adim, HID, HID, 1])
    opt_a = torch.optim.Adam(list(trunk.parameters()) + list(mu_l.parameters())
                             + list(ls_l.parameters()), 3e-4)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 1e-3)
    buf = Replay(sdim, adim)

    def policy(st, deterministic=False):
        h = F.relu(trunk(st))
        mu, log_std = mu_l(h), torch.clamp(ls_l(h), -5, 0)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + std * eps
        a = torch.tanh(pre)
        logp = (-0.5 * ((eps) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
        logp -= torch.log(1 - a ** 2 + 1e-6).sum(-1, keepdim=True)
        return a, logp

    s = env.reset()
    curve = []
    for t in range(TOTAL_STEPS):
        st = torch.as_tensor(s).unsqueeze(0)
        with torch.no_grad():
            a, _ = policy(st)
        a = a.squeeze(0).numpy()
        _, r, _, info = env.step(a)
        buf.add(s, a, r)
        curve.append(info["adr"])

        if buf.n >= 128:
            bs_, ba, br = buf.sample(128)
            qin = torch.cat([bs_, ba], 1)
            loss_c = F.mse_loss(q1(qin), br) + F.mse_loss(q2(qin), br)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

            pa, logp = policy(bs_)
            qv = torch.min(q1(torch.cat([bs_, pa], 1)),
                           q2(torch.cat([bs_, pa], 1)))
            loss_a = (alpha * logp - qv).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()

    with torch.no_grad():
        a_fin, _ = policy(torch.as_tensor(s).unsqueeze(0), deterministic=True)
    return "SAC", np.array(curve), a_fin.squeeze(0).numpy()


# ---------------------------------- PPO -------------------------------------
def run_ppo(env, batch=64, epochs=8, clip=0.2):
    sdim, adim = env.state_dim, env.action_dim
    trunk = mlp([sdim, HID, HID])
    mu_l = nn.Linear(HID, adim)
    log_std = nn.Parameter(torch.zeros(adim) - 0.3)
    vf = mlp([sdim, HID, HID, 1])
    opt = torch.optim.Adam(list(trunk.parameters()) + list(mu_l.parameters())
                           + [log_std] + list(vf.parameters()), 3e-4)

    def dist(st):
        mu = torch.tanh(mu_l(F.relu(trunk(st))))
        return torch.distributions.Normal(mu, log_std.exp())

    s = env.reset()
    curve = []
    steps = 0
    while steps < TOTAL_STEPS:
        S, A, R, LP = [], [], [], []
        for _ in range(batch):
            st = torch.as_tensor(s).unsqueeze(0)
            with torch.no_grad():
                d = dist(st)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
            a_env = np.clip(a.squeeze(0).numpy(), -1, 1)
            _, r, _, info = env.step(a_env)
            S.append(s); A.append(a.squeeze(0).numpy())
            R.append(r); LP.append(lp.item())
            curve.append(info["adr"])
            steps += 1
        S = torch.as_tensor(np.array(S)); A = torch.as_tensor(np.array(A))
        R = torch.as_tensor(np.array(R), dtype=torch.float32).unsqueeze(1)
        LP = torch.as_tensor(np.array(LP), dtype=torch.float32).unsqueeze(1)
        for _ in range(epochs):
            d = dist(S)
            lp = d.log_prob(A).sum(-1, keepdim=True)
            v = vf(S)
            adv = (R - v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            ratio = torch.exp(lp - LP)
            l_pi = -torch.min(ratio * adv,
                              torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            l_v = F.mse_loss(v, R)
            loss = l_pi + 0.5 * l_v - 0.001 * d.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        a_fin = torch.tanh(mu_l(F.relu(trunk(torch.as_tensor(s).unsqueeze(0)))))
    return "PPO", np.array(curve[:TOTAL_STEPS]), a_fin.squeeze(0).numpy()


# ---------------------------------- CAPG ------------------------------------
def run_capg(env):
    """Vanilla continuous-action policy gradient (REINFORCE + baseline)."""
    sdim, adim = env.state_dim, env.action_dim
    trunk = mlp([sdim, HID, HID])
    mu_l = nn.Linear(HID, adim)
    log_std = nn.Parameter(torch.zeros(adim) - 0.3)
    opt = torch.optim.Adam(list(trunk.parameters()) + list(mu_l.parameters())
                           + [log_std], 3e-4)
    s = env.reset()
    baseline, curve = 0.0, []
    for t in range(TOTAL_STEPS):
        st = torch.as_tensor(s).unsqueeze(0)
        mu = torch.tanh(mu_l(F.relu(trunk(st))))
        d = torch.distributions.Normal(mu, log_std.exp())
        a = d.sample()
        _, r, _, info = env.step(np.clip(a.squeeze(0).numpy(), -1, 1))
        curve.append(info["adr"])
        baseline = 0.99 * baseline + 0.01 * r
        loss = -(d.log_prob(a).sum()) * (r - baseline)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a_fin = torch.tanh(mu_l(F.relu(trunk(torch.as_tensor(s).unsqueeze(0)))))
    return "CAPG", np.array(curve), a_fin.squeeze(0).numpy()


# --------------------------------- run all ----------------------------------
if __name__ == "__main__":
    import json, time

    results = {}
    curves = {}
    for fn in [
        lambda e: run_ddpg(e, label="DDPG"),
        run_sac,
        run_ppo,
        run_capg,
        lambda e: run_ddpg(e, twin=True, delayed=True,
                           noise0=0.6, noise_decay=0.999, label="ECDPO"),
    ]:
        env = OIRSEnv()
        t0 = time.time()
        name, curve, a_fin = fn(env)
        gamma, omega = env.angles(a_fin)
        final_adr = adr(gamma, omega)
        results[name] = dict(gamma_deg=float(np.rad2deg(gamma)),
                             omega_deg=float(np.rad2deg(omega)),
                             adr_mbps=float(final_adr / 1e6),
                             train_s=round(time.time() - t0, 1))
        curves[name] = curve
        print(f"{name:6s} gamma={np.rad2deg(gamma):8.3f} deg  "
              f"omega={np.rad2deg(omega):8.3f} deg  ADR={final_adr/1e6:7.3f} Mbit/s  "
              f"({results[name]['train_s']}s)")

    gs, os_, _ = analytic_optimum()
    results["Analytic optimum"] = dict(gamma_deg=float(np.rad2deg(gs)),
                                       omega_deg=float(np.rad2deg(os_)),
                                       adr_mbps=float(adr(gs, os_) / 1e6))
    print(f"{'OPT':6s} gamma={np.rad2deg(gs):8.3f} deg  omega={np.rad2deg(os_):8.3f} deg  "
          f"ADR={adr(gs, os_)/1e6:7.3f} Mbit/s")

    np.savez("curves.npz", **curves)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
