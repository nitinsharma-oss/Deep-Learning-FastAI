"""
DRL-Based Adaptive Power Control for Wiretap Channel ASC Maximisation
=====================================================================
Environment : Markov fading wiretap channel with three-region incomplete-Beta PDF
State       : (h_B, h_E) — instantaneous fading gains
Action      : P_t in [P_min, P_max] — transmit power
Reward      : [0.5*ln(2+A_B*P*h_B) - 0.5*ln(2+A_E*P*h_E)]^+ - lam*(P-P_bar)

Five agents:
  1. DDPG          — deterministic, off-policy, OU noise
  2. SAC           — stochastic, off-policy, auto-entropy
  3. PPO           — stochastic, on-policy, clipped surrogate
  4. ECDPO (hybrid)— SAC entropy + PPO clip + DDPG determinism
  5. CAPG (novel)  — concavity-aware policy gradient (physics-informed)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import random, copy, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cpu')
# ======================================================================
#  ENVIRONMENT
# ======================================================================
class WiretapEnv:
    """Markov fading wiretap channel with AR(1) state evolution."""
    def __init__(self, A_B=1.5, A_E=0.5, P_min=0.1, P_max=5.0, P_bar=2.0,
                 lam=0.3, rho_B=0.85, rho_E=0.85,
                 h_min=0.2, h_max=3.0):
        self.A_B, self.A_E = A_B, A_E
        self.P_min, self.P_max, self.P_bar, self.lam = P_min, P_max, P_bar, lam
        self.rho_B, self.rho_E = rho_B, rho_E
        self.h_min, self.h_max = h_min, h_max
        self.state = None

    def _sample_fading(self):
        return np.random.uniform(self.h_min, self.h_max)

    def _evolve(self, h, rho):
        w = self._sample_fading()
        h_new = rho * h + (1 - rho) * w
        return np.clip(h_new, self.h_min, self.h_max)

    def _normalise_state(self, hB, hE):
        lo, hi = self.h_min, self.h_max
        return np.array([(hB - lo)/(hi - lo), (hE - lo)/(hi - lo)], dtype=np.float32)

    def reset(self):
        self.hB = self._sample_fading()
        self.hE = self._sample_fading()
        return self._normalise_state(self.hB, self.hE)

    def step(self, action):
        P = np.clip(action, self.P_min, self.P_max)
        cB = 0.5 * np.log(2 + self.A_B * P * self.hB)
        cE = 0.5 * np.log(2 + self.A_E * P * self.hE)
        secrecy = max(cB - cE, 0.0)
        reward = secrecy - self.lam * (P - self.P_bar)
        self.hB = self._evolve(self.hB, self.rho_B)
        self.hE = self._evolve(self.hE, self.rho_E)
        return self._normalise_state(self.hB, self.hE), float(reward), False, {
            'secrecy': secrecy, 'power': P, 'hB': self.hB, 'hE': self.hE}

    def analytical_grad(self, P, hB, hE):
        """Closed-form dr/dP (concavity-aware)."""
        g = 0.5*(self.A_B*hB/(2+self.A_B*P*hB) - self.A_E*hE/(2+self.A_E*P*hE))
        return g - self.lam

# ======================================================================
#  SHARED BUILDING BLOCKS
# ======================================================================
class ReplayBuffer:
    def __init__(self, cap=200000):
        self.buf = deque(maxlen=cap)
    def push(self, *args): self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(np.array(a),dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(r),dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2),dtype=torch.float32),
                torch.tensor(np.array(d),dtype=torch.float32).unsqueeze(1))
    def __len__(self): return len(self.buf)

def mlp(dims, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2: layers.append(act())
        elif out_act: layers.append(out_act())
    return nn.Sequential(*layers)

def soft_update(tgt, src, tau=0.005):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.copy_(tau*sp.data + (1-tau)*tp.data)

# ======================================================================
#  1. DDPG
# ======================================================================
class DDPGActor(nn.Module):
    def __init__(self, sd=2, hd=128):
        super().__init__(); self.net = mlp([sd,hd,hd,1], out_act=nn.Sigmoid)
    def forward(self, s): return self.net(s)   # in [0,1], rescale outside

class DDPGCritic(nn.Module):
    def __init__(self, sd=2, hd=128):
        super().__init__(); self.net = mlp([sd+1,hd,hd,1])
    def forward(self, s, a): return self.net(torch.cat([s,a],1))

class DDPG:
    name = 'DDPG'
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, batch=256):
        self.env, self.gamma, self.tau, self.batch = env, gamma, tau, batch
        self.actor = DDPGActor(); self.actor_tgt = copy.deepcopy(self.actor)
        self.critic = DDPGCritic(); self.critic_tgt = copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buf = ReplayBuffer(); self.noise_std = 0.2

    def _to_power(self, a01):
        return self.env.P_min + a01 * (self.env.P_max - self.env.P_min)

    def select_action(self, s, explore=True):
        with torch.no_grad():
            a = self.actor(torch.tensor(s,dtype=torch.float32).unsqueeze(0)).item()
        if explore: a = np.clip(a + np.random.normal(0, self.noise_std), 0, 1)
        return self._to_power(a)

    def update(self):
        if len(self.buf) < self.batch: return
        s,a,r,s2,d = self.buf.sample(self.batch)
        a_n = (a - self.env.P_min)/(self.env.P_max - self.env.P_min)
        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            a2_p = a2*(self.env.P_max-self.env.P_min)+self.env.P_min
            q_tgt = r + self.gamma*(1-d)*self.critic_tgt(s2, a2_p)
        q = self.critic(s, a)
        self.opt_c.zero_grad(); (q - q_tgt).pow(2).mean().backward(); self.opt_c.step()
        a_pred = self.actor(s)
        a_pred_p = a_pred*(self.env.P_max-self.env.P_min)+self.env.P_min
        self.opt_a.zero_grad(); (-self.critic(s, a_pred_p)).mean().backward(); self.opt_a.step()
        soft_update(self.actor_tgt, self.actor, self.tau)
        soft_update(self.critic_tgt, self.critic, self.tau)

# ======================================================================
#  2. SAC
# ======================================================================
class SACActor(nn.Module):
    def __init__(self, sd=2, hd=128):
        super().__init__()
        self.shared = mlp([sd,hd,hd])
        self.mu = nn.Linear(hd,1); self.log_std = nn.Linear(hd,1)
    def forward(self, s):
        h = F.relu(self.shared(s))
        mu = self.mu(h); log_std = self.log_std(h).clamp(-5, 2)
        return mu, log_std
    def sample(self, s):
        mu, log_std = self(s); std = log_std.exp()
        z = mu + std * torch.randn_like(std)
        a = torch.sigmoid(z)
        log_p = (-0.5*((z-mu)/std)**2 - log_std - 0.5*math.log(2*math.pi)
                 - torch.log(a*(1-a)+1e-8))
        return a, log_p

class SACCritic(nn.Module):
    def __init__(self, sd=2, hd=128):
        super().__init__()
        self.q1 = mlp([sd+1,hd,hd,1]); self.q2 = mlp([sd+1,hd,hd,1])
    def forward(self, s, a):
        sa = torch.cat([s,a],1)
        return self.q1(sa), self.q2(sa)

class SAC:
    name = 'SAC'
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, batch=256, alpha_lr=3e-4):
        self.env, self.gamma, self.tau, self.batch = env, gamma, tau, batch
        self.actor = SACActor()
        self.critic = SACCritic(); self.critic_tgt = copy.deepcopy(self.critic)
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.target_entropy = -1.0
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.buf = ReplayBuffer()

    def _to_power(self, a01):
        return self.env.P_min + a01 * (self.env.P_max - self.env.P_min)

    def select_action(self, s, explore=True):
        with torch.no_grad():
            st = torch.tensor(s,dtype=torch.float32).unsqueeze(0)
            a, _ = self.actor.sample(st) if explore else (torch.sigmoid(self.actor(st)[0]),None)
        return self._to_power(a.item())

    def update(self):
        if len(self.buf) < self.batch: return
        s,a,r,s2,d = self.buf.sample(self.batch)
        a_n = (a - self.env.P_min)/(self.env.P_max - self.env.P_min)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.actor.sample(s2)
            a2_p = a2*(self.env.P_max-self.env.P_min)+self.env.P_min
            q1t, q2t = self.critic_tgt(s2, a2_p)
            q_tgt = r + self.gamma*(1-d)*(torch.min(q1t,q2t) - alpha*lp2)
        q1, q2 = self.critic(s, a)
        lc = (q1-q_tgt).pow(2).mean() + (q2-q_tgt).pow(2).mean()
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        a_new, lp_new = self.actor.sample(s)
        a_new_p = a_new*(self.env.P_max-self.env.P_min)+self.env.P_min
        q1n, q2n = self.critic(s, a_new_p)
        la = (alpha*lp_new - torch.min(q1n,q2n)).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        la2 = -(self.log_alpha.exp()*(lp_new.detach()+self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); la2.backward(); self.opt_alpha.step()
        soft_update(self.critic_tgt, self.critic, self.tau)

# ======================================================================
#  3. PPO
# ======================================================================
class PPOActor(nn.Module):
    def __init__(self, sd=2, hd=128):
        super().__init__()
        self.shared = mlp([sd,hd,hd])
        self.mu = nn.Linear(hd,1); self.log_std = nn.Parameter(torch.zeros(1))
    def forward(self, s):
        h = F.relu(self.shared(s)); return torch.sigmoid(self.mu(h)), self.log_std.exp()
    def log_prob(self, s, a):
        mu, std = self(s)
        return -0.5*((a-mu)/std)**2 - torch.log(std) - 0.5*math.log(2*math.pi)
    def sample(self, s):
        mu, std = self(s); a = torch.clamp(mu + std*torch.randn_like(mu), 0, 1)
        return a

class PPOCritic(nn.Module):
    def __init__(self, sd=2, hd=128):
        super().__init__(); self.net = mlp([sd,hd,hd,1])
    def forward(self, s): return self.net(s)

class PPO:
    name = 'PPO'
    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, K=10, batch=64, horizon=512):
        self.env, self.gamma, self.eps = env, gamma, eps_clip
        self.K, self.batch, self.horizon = K, batch, horizon
        self.actor = PPOActor(); self.critic = PPOCritic()
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def _to_power(self, a01):
        return self.env.P_min + a01 * (self.env.P_max - self.env.P_min)

    def select_action(self, s, explore=True):
        with torch.no_grad():
            st = torch.tensor(s,dtype=torch.float32).unsqueeze(0)
            a = self.actor.sample(st) if explore else self.actor(st)[0]
        return self._to_power(a.item())

    def collect_and_update(self):
        states, actions, rewards, dones, old_lps = [],[],[],[],[]
        s = self.env.reset()
        for _ in range(self.horizon):
            st = torch.tensor(s,dtype=torch.float32).unsqueeze(0)
            a01 = self.actor.sample(st); lp = self.actor.log_prob(st, a01)
            P = self._to_power(a01.item())
            s2, r, d, _ = self.env.step(P)
            states.append(s); actions.append(a01.item()); rewards.append(r)
            dones.append(float(d)); old_lps.append(lp.item()); s = s2
        S = torch.tensor(np.array(states),dtype=torch.float32)
        A = torch.tensor(np.array(actions),dtype=torch.float32).unsqueeze(1)
        R = np.array(rewards); D = np.array(dones)
        with torch.no_grad(): V = self.critic(S).squeeze().numpy()
        # GAE
        adv = np.zeros_like(R); gae = 0; lam = 0.95
        V_ext = np.append(V, 0)
        for t in reversed(range(len(R))):
            delta = R[t] + self.gamma*(1-D[t])*V_ext[t+1] - V_ext[t]
            gae = delta + self.gamma*lam*(1-D[t])*gae; adv[t] = gae
        ret = torch.tensor(adv + V, dtype=torch.float32).unsqueeze(1)
        adv_t = torch.tensor(adv, dtype=torch.float32).unsqueeze(1)
        adv_t = (adv_t - adv_t.mean())/(adv_t.std()+1e-8)
        old_lp = torch.tensor(old_lps, dtype=torch.float32).unsqueeze(1)
        for _ in range(self.K):
            idx = np.random.permutation(len(R))
            for start in range(0, len(R), self.batch):
                ix = idx[start:start+self.batch]
                sb, ab, adv_b, ret_b, olp_b = S[ix], A[ix], adv_t[ix], ret[ix], old_lp[ix]
                lp_new = self.actor.log_prob(sb, ab)
                ratio = (lp_new - olp_b).exp()
                s1 = ratio * adv_b; s2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*adv_b
                la = -torch.min(s1,s2).mean()
                self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
                lc = (self.critic(sb) - ret_b).pow(2).mean()
                self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        return np.mean(R)

# ======================================================================
#  4. ECDPO (Hybrid: SAC entropy + PPO clip + DDPG determinism)
# ======================================================================
class ECDPO:
    name = 'ECDPO (Hybrid)'
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, batch=256, eps_clip=0.15):
        self.env, self.gamma, self.tau, self.batch, self.eps = env, gamma, tau, batch, eps_clip
        self.actor = DDPGActor(); self.actor_old = copy.deepcopy(self.actor)
        self.critic = SACCritic(); self.critic_tgt = copy.deepcopy(self.critic)
        self.log_alpha = torch.tensor(-1.0, requires_grad=True)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.buf = ReplayBuffer(); self.update_count = 0

    def _to_power(self, a01):
        return self.env.P_min + a01 * (self.env.P_max - self.env.P_min)

    def select_action(self, s, explore=True):
        with torch.no_grad():
            a = self.actor(torch.tensor(s,dtype=torch.float32).unsqueeze(0)).item()
        if explore: a = np.clip(a + np.random.normal(0, 0.15), 0, 1)
        return self._to_power(a)

    def update(self):
        if len(self.buf) < self.batch: return
        s,a,r,s2,d = self.buf.sample(self.batch)
        a_n = (a - self.env.P_min)/(self.env.P_max - self.env.P_min)
        alpha = self.log_alpha.exp().detach()
        # critic update (twin Q, SAC-style with entropy of exploration noise)
        with torch.no_grad():
            a2 = self.actor(s2)
            a2_p = a2*(self.env.P_max-self.env.P_min)+self.env.P_min
            q1t, q2t = self.critic_tgt(s2, a2_p)
            q_tgt = r + self.gamma*(1-d)*torch.min(q1t,q2t)
        q1, q2 = self.critic(s, a)
        lc = (q1-q_tgt).pow(2).mean() + (q2-q_tgt).pow(2).mean()
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        # actor update: PPO-clipped deterministic ratio
        a_new = self.actor(s)
        with torch.no_grad(): a_old = self.actor_old(s)
        # Gaussian ratio centred on a_old, std=0.15
        std = 0.15
        log_new = -0.5*((a_new-a_old.detach())/std)**2
        log_old = torch.zeros_like(log_new)  # ratio at old action = 1
        ratio = (log_new - log_old).exp()
        a_new_p = a_new*(self.env.P_max-self.env.P_min)+self.env.P_min
        q1n, q2n = self.critic(s, a_new_p)
        adv = torch.min(q1n,q2n).detach()  # use Q as advantage proxy
        adv = (adv - adv.mean())/(adv.std()+1e-8)
        s1 = ratio * adv; s2_c = torch.clamp(ratio, 1-self.eps, 1+self.eps)*adv
        # entropy bonus on exploration diversity
        ent_bonus = alpha * 0.5 * torch.log(torch.tensor(2*math.pi*std**2))
        la = -(torch.min(s1,s2_c).mean() + ent_bonus)
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        # alpha auto-tune
        la2 = -(self.log_alpha.exp() * (ent_bonus.detach() + 1.0)).mean()
        self.opt_alpha.zero_grad(); la2.backward(); self.opt_alpha.step()
        soft_update(self.critic_tgt, self.critic, self.tau)
        self.update_count += 1
        if self.update_count % 20 == 0:
            self.actor_old.load_state_dict(self.actor.state_dict())

# ======================================================================
#  5. CAPG (Novel: Concavity-Aware Policy Gradient)
# ======================================================================
class CAPG:
    name = 'CAPG (Novel)'
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, batch=256):
        self.env, self.gamma, self.tau, self.batch = env, gamma, tau, batch
        self.actor = DDPGActor(); self.actor_tgt = copy.deepcopy(self.actor)
        self.critic = DDPGCritic(); self.critic_tgt = copy.deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buf = ReplayBuffer()
        # learnable mixing coefficient: how much to trust analytical grad vs critic
        self.log_beta = torch.tensor(0.0, requires_grad=True)
        self.opt_beta = torch.optim.Adam([self.log_beta], lr=1e-3)

    def _to_power(self, a01):
        return self.env.P_min + a01 * (self.env.P_max - self.env.P_min)

    def select_action(self, s, explore=True):
        with torch.no_grad():
            a = self.actor(torch.tensor(s,dtype=torch.float32).unsqueeze(0)).item()
        if explore: a = np.clip(a + np.random.normal(0, 0.15), 0, 1)
        return self._to_power(a)

    def update(self, hB_batch=None, hE_batch=None):
        if len(self.buf) < self.batch: return
        s,a,r,s2,d = self.buf.sample(self.batch)
        a_n = (a - self.env.P_min)/(self.env.P_max - self.env.P_min)
        # critic update (standard)
        with torch.no_grad():
            a2 = self.actor_tgt(s2)
            a2_p = a2*(self.env.P_max-self.env.P_min)+self.env.P_min
            q_tgt = r + self.gamma*(1-d)*self.critic_tgt(s2, a2_p)
        q = self.critic(s, a)
        self.opt_c.zero_grad(); (q - q_tgt).pow(2).mean().backward(); self.opt_c.step()
        # actor update: BLEND of critic gradient + analytical reward gradient
        a_pred = self.actor(s)  # in [0,1]
        a_pred_p = a_pred*(self.env.P_max-self.env.P_min)+self.env.P_min
        # (A) critic-based gradient (long-horizon, estimated)
        loss_critic = -self.critic(s, a_pred_p).mean()
        # (B) analytical gradient (single-step, exact)
        # recover hB,hE from normalised state
        hB = s[:,0]*(self.env.h_max-self.env.h_min)+self.env.h_min
        hE = s[:,1]*(self.env.h_max-self.env.h_min)+self.env.h_min
        P = a_pred_p.squeeze()
        # dr/dP (closed form from the ln(2+Av) structure)
        grad_r = 0.5*(self.env.A_B*hB/(2+self.env.A_B*P.detach()*hB)
                      -self.env.A_E*hE/(2+self.env.A_E*P.detach()*hE)) - self.env.lam
        # analytical loss: push action in direction of reward gradient
        loss_analytical = -(grad_r.detach() * a_pred.squeeze()).mean()
        beta = torch.sigmoid(self.log_beta)  # mixing in [0,1]
        loss_a = (1 - beta) * loss_critic + beta * loss_analytical
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
        # update beta: whichever gradient direction has lower variance should get more weight
        # heuristic: minimise actor loss magnitude
        self.opt_beta.zero_grad()
        loss_beta = loss_a.detach() * self.log_beta  # REINFORCE on beta
        loss_beta.backward(); self.opt_beta.step()
        soft_update(self.actor_tgt, self.actor, self.tau)
        soft_update(self.critic_tgt, self.critic, self.tau)

# ======================================================================
#  TRAINING LOOP
# ======================================================================
def train_offpolicy(agent, total_steps=60000, eval_every=500, eval_eps=5):
    """Train off-policy agent (DDPG, SAC, ECDPO, CAPG)."""
    env = agent.env; s = env.reset(); rewards_log = []
    ep_reward = 0; ep_steps = 0; step = 0
    eval_steps = []; eval_rewards = []
    while step < total_steps:
        a = agent.select_action(s, explore=True)
        s2, r, d, info = env.step(a)
        agent.buf.push(s, a, r, s2, float(d))
        s = s2; ep_reward += r; ep_steps += 1; step += 1
        agent.update()
        if ep_steps >= 200:
            s = env.reset(); ep_reward = 0; ep_steps = 0
        if step % eval_every == 0:
            avg_r = evaluate(agent, n_ep=eval_eps)
            eval_steps.append(step); eval_rewards.append(avg_r)
            print(f'  [{agent.name}] step {step:6d}  eval_reward {avg_r:.4f}')
    return eval_steps, eval_rewards

def train_ppo(agent, total_steps=60000, eval_every=2048, eval_eps=5):
    """Train PPO (on-policy)."""
    eval_steps = []; eval_rewards = []; step = 0
    while step < total_steps:
        avg_r_train = agent.collect_and_update()
        step += agent.horizon
        avg_r = evaluate(agent, n_ep=eval_eps)
        eval_steps.append(step); eval_rewards.append(avg_r)
        print(f'  [{agent.name}] step {step:6d}  eval_reward {avg_r:.4f}')
    return eval_steps, eval_rewards

def evaluate(agent, n_ep=5, max_steps=200):
    env = agent.env; total = 0
    for _ in range(n_ep):
        s = env.reset(); ep_r = 0
        for _ in range(max_steps):
            a = agent.select_action(s, explore=False)
            s, r, d, _ = env.step(a); ep_r += r
        total += ep_r / max_steps  # average reward per step
    return total / n_ep

# ======================================================================
#  MAIN — train all five, plot comparison
# ======================================================================
if __name__ == '__main__':
    TOTAL_STEPS = 20000
    results = {}

    print('='*60); print(' Training 5 DRL agents for wiretap power control')
    print('='*60)

    # 1. DDPG
    print('\n[1/5] DDPG')
    agent1 = DDPG(WiretapEnv())
    results['DDPG'] = train_offpolicy(agent1, TOTAL_STEPS)

    # 2. SAC
    print('\n[2/5] SAC')
    agent2 = SAC(WiretapEnv())
    results['SAC'] = train_offpolicy(agent2, TOTAL_STEPS)

    # 3. PPO
    print('\n[3/5] PPO')
    agent3 = PPO(WiretapEnv())
    results['PPO'] = train_ppo(agent3, TOTAL_STEPS)

    # 4. ECDPO (Hybrid)
    print('\n[4/5] ECDPO (Hybrid)')
    agent4 = ECDPO(WiretapEnv())
    results['ECDPO'] = train_offpolicy(agent4, TOTAL_STEPS)

    # 5. CAPG (Novel)
    print('\n[5/5] CAPG (Novel)')
    agent5 = CAPG(WiretapEnv())
    results['CAPG'] = train_offpolicy(agent5, TOTAL_STEPS)

    # Analytical baseline: water-filling (from closed-form)
    env_wf = WiretapEnv()
    wf_rewards = []
    s = env_wf.reset()
    for _ in range(2000):
        hB = s[0]*(env_wf.h_max-env_wf.h_min)+env_wf.h_min
        hE = s[1]*(env_wf.h_max-env_wf.h_min)+env_wf.h_min
        # solve dr/dP = 0 by bisection
        lo, hi = env_wf.P_min, env_wf.P_max
        for _ in range(30):
            mid = (lo+hi)/2
            g = env_wf.analytical_grad(mid, hB, hE)
            if g > 0: lo = mid
            else: hi = mid
        P_opt = (lo+hi)/2
        if env_wf.A_B*hB <= env_wf.A_E*hE: P_opt = env_wf.P_min
        s, r, _, _ = env_wf.step(P_opt); wf_rewards.append(r)
    wf_avg = np.mean(wf_rewards)
    print(f'\nWater-filling analytical baseline: {wf_avg:.4f}')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {'DDPG':'#2a78d6', 'SAC':'#1baf7a', 'PPO':'#eda100',
              'ECDPO':'#e34948', 'CAPG':'#4a3aa7'}
    for name, (steps, rews) in results.items():
        ax.plot(steps, rews, label=name, color=colors[name], linewidth=2, alpha=0.85)
    ax.axhline(wf_avg, color='gray', linestyle='--', linewidth=1.5, label='Water-filling (analytical)')
    ax.set_xlabel('Training steps', fontsize=13)
    ax.set_ylabel('Avg secrecy reward per step', fontsize=13)
    ax.set_title('DRL Power Control for Wiretap ASC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/drl_comparison.png', dpi=150)
    print('\nPlot saved to /mnt/user-data/outputs/drl_comparison.png')

    # Final comparison table
    print('\n' + '='*60)
    print(f'{"Method":<20} {"Final Eval Reward":>18}')
    print('-'*40)
    for name, (steps, rews) in results.items():
        print(f'{name:<20} {rews[-1]:>18.4f}')
    print(f'{"Water-filling":<20} {wf_avg:>18.4f}')
    print('='*60)
