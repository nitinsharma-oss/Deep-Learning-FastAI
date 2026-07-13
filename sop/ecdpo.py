"""
ECDPO (Hybrid) — Entropy-regularised Clipped Deterministic Policy Optimization
Combines: DDPG's deterministic actor + SAC's twin-Q critics + PPO-inspired
          entropy regularisation for exploration stability.
v_B = b^2*Pt*gamma_B*Po, v_E = b^2*Pt*gamma_E*Po, rate = log2(1+A*v)
Run:  python ecdpo.py
"""
import numpy as np, torch, torch.nn as nn, random, copy
from collections import deque
from wiretap_env import WiretapEnv, evaluate

H, BATCH, GAMMA, TAU, LR = 64, 128, 0.99, 0.005, 3e-4
ENT_COEF = 0.05        # entropy bonus weight (from SAC's idea, fixed here)

def mlp(dims, out_act=None):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2: layers.append(nn.ReLU())
    if out_act: layers.append(out_act())
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

class ECDPO:
    """
    Hybrid design:
      * deterministic sigmoid actor (DDPG) -> precise power at deployment
      * twin Q-networks with min-target  (SAC/TD3) -> no overestimation bias
      * Bernoulli-entropy bonus on actor output (PPO/SAC spirit) -> keeps the
        policy away from saturated boundary actions during learning
    """
    def __init__(self, env):
        self.env = env
        self.actor = mlp([2,H,H,1], nn.Sigmoid)
        self.q1 = mlp([3,H,H,1]); self.q2 = mlp([3,H,H,1])
        self.q1_t = copy.deepcopy(self.q1); self.q2_t = copy.deepcopy(self.q2)
        self.oa = torch.optim.Adam(self.actor.parameters(), LR)
        self.oq = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), LR)
        self.buf = Buffer(); self.noise = 0.15

    def to_P(self, a01):
        return self.env.P_min + a01*(self.env.P_max - self.env.P_min)

    def act(self, s, explore=True):
        with torch.no_grad():
            a = self.actor(torch.tensor(s).unsqueeze(0)).item()
        if explore: a = float(np.clip(a + np.random.normal(0, self.noise), 0, 1))
        return self.to_P(a)

    def update(self):
        if len(self.buf) < BATCH: return
        S, A, R, S2 = self.buf.sample(BATCH)
        # --- twin-Q critic update (min of targets)
        with torch.no_grad():
            a2 = self.actor(S2)
            sa2 = torch.cat([S2, self.to_P(a2)], 1)
            q_tgt = R + GAMMA*torch.min(self.q1_t(sa2), self.q2_t(sa2))
        sa = torch.cat([S, A], 1)
        lq = (self.q1(sa)-q_tgt).pow(2).mean() + (self.q2(sa)-q_tgt).pow(2).mean()
        self.oq.zero_grad(); lq.backward(); self.oq.step()
        # --- actor: maximise min-Q + entropy of the (0,1) action
        a_new = self.actor(S)
        sa_new = torch.cat([S, self.to_P(a_new)], 1)
        q_min = torch.min(self.q1(sa_new), self.q2(sa_new))
        entropy = -(a_new*torch.log(a_new + 1e-8)
                    + (1-a_new)*torch.log(1-a_new + 1e-8)).mean()
        la = -q_min.mean() - ENT_COEF*entropy
        self.oa.zero_grad(); la.backward(); self.oa.step()
        soft(self.q1_t, self.q1); soft(self.q2_t, self.q2)

def train(steps=10000, eval_every=1000):
    env = WiretapEnv(); agent = ECDPO(env)
    s = env.reset(); ep = 0
    for t in range(1, steps+1):
        P = agent.act(s)
        s2, r, _, _ = env.step(P)
        agent.buf.push(s, P, r, s2); s = s2; ep += 1
        if ep >= 200: s = env.reset(); ep = 0
        agent.update()
        if t % eval_every == 0:
            avg = evaluate(lambda st: agent.act(st, explore=False), env)
            print(f'[ECDPO] step {t:6d}  eval avg reward/step = {avg:.4f}')
    return agent

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    agent = train()
    env = agent.env; wf = evaluate(lambda st: env.waterfill_P(*env.denorm(np.asarray(st))), env, n_ep=10)
    print(f'[ECDPO] water-filling baseline = {wf:.4f}')
