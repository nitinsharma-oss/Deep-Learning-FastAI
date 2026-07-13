"""
DDPG — Deep Deterministic Policy Gradient for wiretap power control
v_B = b^2*Pt*gamma_B*Po, v_E = b^2*Pt*gamma_E*Po, rate = log2(1+A*v)
Run:  python ddpg.py
"""
import numpy as np, torch, torch.nn as nn, random, copy
from collections import deque
from wiretap_env import WiretapEnv, evaluate, plot_learning_curve

H, BATCH, GAMMA, TAU, LR = 64, 128, 0.99, 0.005, 3e-4

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

class DDPG:
    def __init__(self, env):
        self.env = env
        self.actor  = mlp([2,H,H,1], nn.Sigmoid);  self.actor_t  = copy.deepcopy(self.actor)
        self.critic = mlp([3,H,H,1]);              self.critic_t = copy.deepcopy(self.critic)
        self.oa = torch.optim.Adam(self.actor.parameters(),  LR)
        self.oc = torch.optim.Adam(self.critic.parameters(), LR)
        self.buf = Buffer(); self.noise = 0.2

    def to_P(self, a01):     # map sigmoid output -> [P_min, P_max]
        return self.env.P_min + a01*(self.env.P_max - self.env.P_min)

    def act(self, s, explore=True):
        with torch.no_grad():
            a = self.actor(torch.tensor(s).unsqueeze(0)).item()
        if explore: a = float(np.clip(a + np.random.normal(0, self.noise), 0, 1))
        return self.to_P(a)

    def update(self):
        if len(self.buf) < BATCH: return
        S, A, R, S2 = self.buf.sample(BATCH)
        with torch.no_grad():
            a2 = self.actor_t(S2)
            q_tgt = R + GAMMA*self.critic_t(torch.cat([S2, self.to_P(a2)], 1))
        q = self.critic(torch.cat([S, A], 1))
        self.oc.zero_grad(); (q - q_tgt).pow(2).mean().backward(); self.oc.step()
        a_pred = self.actor(S)
        self.oa.zero_grad()
        (-self.critic(torch.cat([S, self.to_P(a_pred)], 1))).mean().backward()
        self.oa.step()
        soft(self.actor_t, self.actor); soft(self.critic_t, self.critic)

def train(steps=10000, eval_every=1000):
    env = WiretapEnv(); agent = DDPG(env)
    s = env.reset(); ep = 0; log = []
    for t in range(1, steps+1):
        P = agent.act(s)
        s2, r, _, _ = env.step(P)
        agent.buf.push(s, P, r, s2); s = s2; ep += 1
        if ep >= 200: s = env.reset(); ep = 0
        agent.update()
        if t % eval_every == 0:
            avg = evaluate(lambda st: agent.act(st, explore=False), env)
            log.append((t, avg))
            print(f'[DDPG] step {t:6d}  eval avg reward/step = {avg:.4f}')
    return agent, log

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    agent, log = train()
    # compare with analytical water-filling
    env = agent.env; wf = evaluate(lambda st: env.waterfill_P(*env.denorm(np.asarray(st))), env, n_ep=10)
    plot_learning_curve(log, 'DDPG', wf=wf, color='#2a78d6')
    print(f'[DDPG] water-filling baseline = {wf:.4f}')
