"""
CAPG (Novel) — Concavity-Aware Policy Gradient for wiretap power control
Physics-informed: the reward gradient w.r.t. power is available in CLOSED FORM
from the log2(1+A*v) structure with v = b^2*Pt*gamma*Po:

  dr/dP = (1/ln2) * [ A_B*c*gB/(1+A_B*c*P*gB) - A_E*c*gE/(1+A_E*c*P*gE) ] - lam
  (c = b^2*Po)

The actor loss BLENDS the critic gradient (long-horizon, learned) with this
exact analytical gradient (single-step, exact) via a learnable coefficient beta.
Run:  python capg.py
"""
import numpy as np, torch, torch.nn as nn, random, copy, math
from collections import deque
from wiretap_env import WiretapEnv, evaluate

H, BATCH, GAMMA, TAU, LR = 64, 128, 0.99, 0.005, 3e-4
LOG2 = math.log(2.0)

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

class CAPG:
    def __init__(self, env):
        self.env = env
        self.actor  = mlp([2,H,H,1], nn.Sigmoid);  self.actor_t  = copy.deepcopy(self.actor)
        self.critic = mlp([3,H,H,1]);              self.critic_t = copy.deepcopy(self.critic)
        self.oa = torch.optim.Adam(self.actor.parameters(),  LR)
        self.oc = torch.optim.Adam(self.critic.parameters(), LR)
        # learnable blend: beta = sigmoid(log_beta) in (0,1)
        # beta -> 1: trust analytical gradient; beta -> 0: trust critic
        self.log_beta = torch.tensor(0.0, requires_grad=True)
        self.ob = torch.optim.Adam([self.log_beta], 1e-3)
        self.buf = Buffer(); self.noise = 0.15

    def to_P(self, a01):
        return self.env.P_min + a01*(self.env.P_max - self.env.P_min)

    def act(self, s, explore=True):
        with torch.no_grad():
            a = self.actor(torch.tensor(s).unsqueeze(0)).item()
        if explore: a = float(np.clip(a + np.random.normal(0, self.noise), 0, 1))
        return self.to_P(a)

    def analytical_dr_dP(self, S, P):
        """Exact dr/dP for the batch, from the log2(1+Av) closed form."""
        e = self.env
        gB = S[:, 0]*(e.g_max - e.g_min) + e.g_min      # de-normalise gamma_B
        gE = S[:, 1]*(e.g_max - e.g_min) + e.g_min      # de-normalise gamma_E
        tB = e.A_B * e.c * gB                            # A_B * b^2 * Po * gamma_B
        tE = e.A_E * e.c * gE
        return (tB/(1.0 + tB*P) - tE/(1.0 + tE*P))/LOG2 - e.lam

    def update(self):
        if len(self.buf) < BATCH: return
        S, A, R, S2 = self.buf.sample(BATCH)
        # --- critic update (standard TD)
        with torch.no_grad():
            a2 = self.actor_t(S2)
            q_tgt = R + GAMMA*self.critic_t(torch.cat([S2, self.to_P(a2)], 1))
        q = self.critic(torch.cat([S, A], 1))
        self.oc.zero_grad(); (q - q_tgt).pow(2).mean().backward(); self.oc.step()
        # --- actor update: blend of critic gradient + analytical gradient
        a_pred = self.actor(S)                           # in (0,1)
        P_pred = self.to_P(a_pred)
        # (A) critic-based loss (long-horizon estimate)
        loss_critic = -self.critic(torch.cat([S, P_pred], 1)).mean()
        # (B) analytical loss: push each action along the EXACT reward gradient
        grad_r = self.analytical_dr_dP(S, P_pred.squeeze().detach())
        loss_analytical = -(grad_r.detach() * a_pred.squeeze()).mean()
        beta = torch.sigmoid(self.log_beta)
        loss_actor = (1-beta)*loss_critic + beta*loss_analytical
        self.oa.zero_grad(); loss_actor.backward(); self.oa.step()
        # --- adapt beta (REINFORCE-style on the blended loss magnitude)
        self.ob.zero_grad()
        (loss_actor.detach()*self.log_beta).backward()
        self.ob.step()
        soft(self.actor_t, self.actor); soft(self.critic_t, self.critic)

def train(steps=10000, eval_every=1000):
    env = WiretapEnv(); agent = CAPG(env)
    s = env.reset(); ep = 0
    for t in range(1, steps+1):
        P = agent.act(s)
        s2, r, _, _ = env.step(P)
        agent.buf.push(s, P, r, s2); s = s2; ep += 1
        if ep >= 200: s = env.reset(); ep = 0
        agent.update()
        if t % eval_every == 0:
            avg = evaluate(lambda st: agent.act(st, explore=False), env)
            beta = torch.sigmoid(agent.log_beta).item()
            print(f'[CAPG] step {t:6d}  eval avg reward/step = {avg:.4f}  (beta={beta:.2f})')
    return agent

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    agent = train()
    env = agent.env; wf = evaluate(lambda st: env.waterfill_P(*env.denorm(np.asarray(st))), env, n_ep=10)
    print(f'[CAPG] water-filling baseline = {wf:.4f}')
