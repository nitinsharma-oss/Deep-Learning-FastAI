"""
DDPG — Deep Deterministic Policy Gradient for wiretap power control
v_B = b^2*Pt*gamma_B*Po, v_E = b^2*Pt*gamma_E*Po, rate = log2(1+A*v)
Trained for 150,000 steps with noise decay and 20-episode evaluation.
Run:  python ddpg.py
"""
import numpy as np, torch, torch.nn as nn, random, copy, time
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
    def __init__(self, cap=200000): self.b = deque(maxlen=cap)
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
        self.buf = Buffer()

    def to_P(self, a01):
        return self.env.P_min + a01*(self.env.P_max - self.env.P_min)

    def act(self, s, noise_std=0.0):
        with torch.no_grad():
            a = self.actor(torch.tensor(s).unsqueeze(0)).item()
        if noise_std > 0:
            a = float(np.clip(a + np.random.normal(0, noise_std), 0, 1))
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

def train(steps=150000, eval_every=5000, eval_eps=20,
          noise_start=0.2, noise_end=0.02, noise_decay_frac=0.7):
    """
    Train DDPG for `steps` environment steps.
    Exploration noise decays linearly from noise_start to noise_end
    over the first noise_decay_frac fraction of training.
    Evaluation uses `eval_eps` episodes with fixed seed for smoothness.
    """
    env = WiretapEnv(); agent = DDPG(env)
    s = env.reset(); ep_steps = 0; log = []
    noise_decay_steps = int(steps * noise_decay_frac)
    t0 = time.time()
    print(f'[DDPG] starting {steps:,} steps  (eval every {eval_every:,}, {eval_eps} eps/eval)')
    for t in range(1, steps+1):
        # linear noise decay
        if t <= noise_decay_steps:
            noise = noise_start + (noise_end - noise_start)*(t / noise_decay_steps)
        else:
            noise = noise_end
        P = agent.act(s, noise_std=noise)
        s2, r, _, _ = env.step(P)
        agent.buf.push(s, P, r, s2); s = s2; ep_steps += 1
        if ep_steps >= 200: s = env.reset(); ep_steps = 0
        agent.update()
        if t % eval_every == 0:
            # fixed-seed evaluation for smoother curves
            rng_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            np.random.seed(42); torch.manual_seed(42)
            avg = evaluate(lambda st: agent.act(st, noise_std=0.0), env, n_ep=eval_eps)
            np.random.set_state(rng_state)
            torch.random.set_rng_state(torch_state)
            log.append((t, avg))
            elapsed = time.time() - t0
            eta = elapsed / t * (steps - t)
            print(f'[DDPG] step {t:>7,}/{steps:,}  reward={avg:.4f}  '
                  f'noise={noise:.3f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s')
    print(f'[DDPG] training complete in {time.time()-t0:.1f}s')
    return agent, log

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    agent, log = train()
    # compare with analytical water-filling (also fixed seed)
    np.random.seed(42); torch.manual_seed(42)
    env = agent.env
    wf = evaluate(lambda st: env.waterfill_P(*env.denorm(np.asarray(st))), env, n_ep=20)
    plot_learning_curve(log, 'DDPG', wf=wf, color='#2a78d6')
    print(f'[DDPG] water-filling baseline = {wf:.4f}')
    print(f'[DDPG] final eval reward      = {log[-1][1]:.4f}')
