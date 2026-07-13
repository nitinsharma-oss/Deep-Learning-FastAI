"""
SAC — Soft Actor-Critic for wiretap power control
v_B = b^2*Pt*gamma_B*Po, v_E = b^2*Pt*gamma_E*Po, rate = log2(1+A*v)
Run:  python sac.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, copy, math
from collections import deque
from wiretap_env import WiretapEnv, evaluate

H, BATCH, GAMMA, TAU, LR = 64, 128, 0.99, 0.005, 3e-4
TARGET_ENTROPY = -1.0

def mlp(dims):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2: layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class Buffer:
    def __init__(self, cap=80000): self.b = deque(maxlen=cap)
    def push(self, *tr): self.b.append(tr)
    def sample(self, n):
        s,a,r,s2 = zip(*random.sample(self.b, n))
        f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
        return f(s), f(a).unsqueeze(1), f(r).unsqueeze(1), f(s2)
    def __len__(self): return len(self.b)

def soft(tgt, src):
    for t,s in zip(tgt.parameters(), src.parameters()):
        t.data.copy_(TAU*s.data + (1-TAU)*t.data)

class GaussianActor(nn.Module):
    """Squashed Gaussian: outputs action in (0,1) with log-prob correction."""
    def __init__(self):
        super().__init__()
        self.shared = mlp([2, H, H])
        self.mu = nn.Linear(H, 1); self.log_std = nn.Linear(H, 1)
    def forward(self, s):
        h = F.relu(self.shared(s))
        return self.mu(h), self.log_std(h).clamp(-5, 2)
    def sample(self, s):
        mu, log_std = self(s); std = log_std.exp()
        z = mu + std*torch.randn_like(std)
        a = torch.sigmoid(z)
        log_p = (-0.5*((z-mu)/std)**2 - log_std - 0.5*math.log(2*math.pi)
                 - torch.log(a*(1-a) + 1e-8))       # sigmoid squash correction
        return a, log_p
    def mean_action(self, s):
        return torch.sigmoid(self(s)[0])

class SAC:
    def __init__(self, env):
        self.env = env
        self.actor = GaussianActor()
        self.q1 = mlp([3,H,H,1]); self.q2 = mlp([3,H,H,1])
        self.q1_t = copy.deepcopy(self.q1); self.q2_t = copy.deepcopy(self.q2)
        self.log_alpha = torch.tensor(0.0, requires_grad=True)   # auto-tuned temperature
        self.oa = torch.optim.Adam(self.actor.parameters(), LR)
        self.oq = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), LR)
        self.ot = torch.optim.Adam([self.log_alpha], LR)
        self.buf = Buffer()

    def to_P(self, a01):
        return self.env.P_min + a01*(self.env.P_max - self.env.P_min)

    def act(self, s, explore=True):
        with torch.no_grad():
            st = torch.tensor(s).unsqueeze(0)
            a = self.actor.sample(st)[0] if explore else self.actor.mean_action(st)
        return self.to_P(a.item())

    def update(self):
        if len(self.buf) < BATCH: return
        S, A, R, S2 = self.buf.sample(BATCH)
        alpha = self.log_alpha.exp().detach()
        # --- twin critic update with entropy-regularised target
        with torch.no_grad():
            a2, lp2 = self.actor.sample(S2)
            sa2 = torch.cat([S2, self.to_P(a2)], 1)
            q_tgt = R + GAMMA*(torch.min(self.q1_t(sa2), self.q2_t(sa2)) - alpha*lp2)
        sa = torch.cat([S, A], 1)
        lq = (self.q1(sa)-q_tgt).pow(2).mean() + (self.q2(sa)-q_tgt).pow(2).mean()
        self.oq.zero_grad(); lq.backward(); self.oq.step()
        # --- actor update (reparameterised)
        a_new, lp = self.actor.sample(S)
        sa_new = torch.cat([S, self.to_P(a_new)], 1)
        la = (alpha*lp - torch.min(self.q1(sa_new), self.q2(sa_new))).mean()
        self.oa.zero_grad(); la.backward(); self.oa.step()
        # --- temperature auto-tuning
        lt = -(self.log_alpha.exp()*(lp.detach() + TARGET_ENTROPY)).mean()
        self.ot.zero_grad(); lt.backward(); self.ot.step()
        soft(self.q1_t, self.q1); soft(self.q2_t, self.q2)

def train(steps=10000, eval_every=1000):
    env = WiretapEnv(); agent = SAC(env)
    s = env.reset(); ep = 0
    for t in range(1, steps+1):
        P = agent.act(s)
        s2, r, _, _ = env.step(P)
        agent.buf.push(s, P, r, s2); s = s2; ep += 1
        if ep >= 200: s = env.reset(); ep = 0
        agent.update()
        if t % eval_every == 0:
            avg = evaluate(lambda st: agent.act(st, explore=False), env)
            print(f'[SAC ] step {t:6d}  eval avg reward/step = {avg:.4f}')
    return agent

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    agent = train()
    env = agent.env; wf = evaluate(lambda st: env.waterfill_P(*env.denorm(np.asarray(st))), env, n_ep=10)
    print(f'[SAC ] water-filling baseline = {wf:.4f}')
