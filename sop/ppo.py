"""
PPO — Proximal Policy Optimization for wiretap power control
v_B = b^2*Pt*gamma_B*Po, v_E = b^2*Pt*gamma_E*Po, rate = log2(1+A*v)
Run:  python ppo.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, math
from wiretap_env import WiretapEnv, evaluate, plot_learning_curve

H, LR, GAMMA, LAM_GAE = 64, 3e-4, 0.99, 0.95
CLIP, EPOCHS, MB, HORIZON = 0.2, 8, 64, 256

def mlp(dims):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2: layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class GaussianActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = mlp([2, H, H])
        self.mu = nn.Linear(H, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # state-independent std
    def forward(self, s):
        h = F.relu(self.shared(s))
        return torch.sigmoid(self.mu(h))
    def log_prob(self, s, a):
        mu = self(s); std = self.log_std.exp()
        return -0.5*((a-mu)/std)**2 - torch.log(std) - 0.5*math.log(2*math.pi)
    def sample(self, s):
        mu = self(s); std = self.log_std.exp()
        return torch.clamp(mu + std*torch.randn_like(mu), 0.0, 1.0)

class PPO:
    def __init__(self, env):
        self.env = env
        self.actor = GaussianActor(); self.valuef = mlp([2, H, H, 1])
        self.oa = torch.optim.Adam(self.actor.parameters(),  LR)
        self.ov = torch.optim.Adam(self.valuef.parameters(), LR)

    def to_P(self, a01):
        return self.env.P_min + a01*(self.env.P_max - self.env.P_min)

    def act(self, s, explore=True):
        with torch.no_grad():
            st = torch.tensor(s).unsqueeze(0)
            a = self.actor.sample(st) if explore else self.actor(st)
        return self.to_P(a.item())

    def rollout_and_update(self):
        # ---- collect on-policy rollout
        S_, A_, R_, LP_ = [], [], [], []
        s = self.env.reset()
        for _ in range(HORIZON):
            st = torch.tensor(s).unsqueeze(0)
            a = self.actor.sample(st)
            lp = self.actor.log_prob(st, a)
            s2, r, _, _ = self.env.step(self.to_P(a.item()))
            S_.append(s); A_.append(a.item()); R_.append(r); LP_.append(lp.item())
            s = s2
        S  = torch.tensor(np.array(S_),  dtype=torch.float32)
        A  = torch.tensor(A_,  dtype=torch.float32).unsqueeze(1)
        R  = np.asarray(R_)
        OL = torch.tensor(LP_, dtype=torch.float32).unsqueeze(1)
        # ---- GAE advantages
        with torch.no_grad(): V = self.valuef(S).squeeze().numpy()
        adv = np.zeros_like(R); gae = 0.0; V_ext = np.append(V, 0.0)
        for t in reversed(range(len(R))):
            delta = R[t] + GAMMA*V_ext[t+1] - V_ext[t]
            gae = delta + GAMMA*LAM_GAE*gae
            adv[t] = gae
        RET = torch.tensor(adv + V, dtype=torch.float32).unsqueeze(1)
        ADV = torch.tensor(adv, dtype=torch.float32).unsqueeze(1)
        ADV = (ADV - ADV.mean())/(ADV.std() + 1e-8)
        # ---- K epochs of clipped-surrogate minibatch updates
        for _ in range(EPOCHS):
            idx = np.random.permutation(len(R))
            for i in range(0, len(R), MB):
                j = idx[i:i+MB]
                sb, ab, advb, retb, olpb = S[j], A[j], ADV[j], RET[j], OL[j]
                lp = self.actor.log_prob(sb, ab)
                ratio = (lp - olpb).exp()
                surr1 = ratio*advb
                surr2 = torch.clamp(ratio, 1-CLIP, 1+CLIP)*advb
                la = -torch.min(surr1, surr2).mean()
                self.oa.zero_grad(); la.backward(); self.oa.step()
                lv = (self.valuef(sb) - retb).pow(2).mean()
                self.ov.zero_grad(); lv.backward(); self.ov.step()
        return float(np.mean(R))

def train(steps=10000, eval_every=1024):
    env = WiretapEnv(); agent = PPO(env)
    done = 0; log = []
    while done < steps:
        agent.rollout_and_update()
        done += HORIZON
        if done % eval_every < HORIZON:
            avg = evaluate(lambda st: agent.act(st, explore=False), env)
            log.append((done, avg))
            print(f'[PPO ] step {done:6d}  eval avg reward/step = {avg:.4f}')
    return agent, log

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    agent, log = train()
    env = agent.env; wf = evaluate(lambda st: env.waterfill_P(*env.denorm(np.asarray(st))), env, n_ep=10)
    plot_learning_curve(log, 'PPO', wf=wf, color='#eda100')
    print(f'[PPO ] water-filling baseline = {wf:.4f}')
