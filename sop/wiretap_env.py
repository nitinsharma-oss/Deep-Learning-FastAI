"""
Shared Wiretap Channel Environment
==================================
SNR relation :  v_B = b^2 * P_t * gamma_B * P_o
                v_E = b^2 * P_t * gamma_E * P_o
Rate         :  C = log2(1 + A * v)
Reward       :  r = [log2(1+A_B*v_B) - log2(1+A_E*v_E)]^+  -  lam*(P_t - P_bar)
State        :  (gamma_B, gamma_E) normalized to [0,1]^2, AR(1) Markov evolution
Action       :  P_t in [P_min, P_max]  (transmit power)
"""
import numpy as np

LOG2 = np.log(2.0)

class WiretapEnv:
    def __init__(self, A_B=1.5, A_E=0.5,          # constant coefficients, A_B > A_E
                 b=1.0, P_o=1.0,                   # optical constants: v = b^2 * P * gamma * P_o
                 P_min=0.1, P_max=5.0, P_bar=2.0,  # power range and average budget
                 lam=0.3,                          # Lagrange penalty on power
                 rho_B=0.85, rho_E=0.85,           # AR(1) fading correlation
                 g_min=0.2, g_max=3.0):            # fading gain support
        assert A_B > A_E > 0, "design constraint A_B > A_E violated"
        self.A_B, self.A_E = A_B, A_E
        self.c = b**2 * P_o                        # combined constant: v = c * P * gamma
        self.P_min, self.P_max, self.P_bar, self.lam = P_min, P_max, P_bar, lam
        self.rho_B, self.rho_E = rho_B, rho_E
        self.g_min, self.g_max = g_min, g_max

    # ---- state helpers -------------------------------------------------
    def _norm(self):
        d = self.g_max - self.g_min
        return np.array([(self.gB - self.g_min)/d,
                         (self.gE - self.g_min)/d], dtype=np.float32)

    def denorm(self, s):
        """Recover (gamma_B, gamma_E) from normalized state (works on tensors too)."""
        d = self.g_max - self.g_min
        return s[..., 0]*d + self.g_min, s[..., 1]*d + self.g_min

    # ---- gym-like API ---------------------------------------------------
    def reset(self):
        self.gB = np.random.uniform(self.g_min, self.g_max)
        self.gE = np.random.uniform(self.g_min, self.g_max)
        return self._norm()

    def step(self, action):
        P = float(np.clip(action, self.P_min, self.P_max))
        vB = self.c * P * self.gB                  # v_B = b^2 * P_t * gamma_B * P_o
        vE = self.c * P * self.gE                  # v_E = b^2 * P_t * gamma_E * P_o
        cB = np.log2(1.0 + self.A_B * vB)
        cE = np.log2(1.0 + self.A_E * vE)
        secrecy = max(cB - cE, 0.0)
        reward = secrecy - self.lam * (P - self.P_bar)
        # AR(1) Markov fading evolution
        self.gB = np.clip(self.rho_B*self.gB + (1-self.rho_B)*np.random.uniform(self.g_min, self.g_max),
                          self.g_min, self.g_max)
        self.gE = np.clip(self.rho_E*self.gE + (1-self.rho_E)*np.random.uniform(self.g_min, self.g_max),
                          self.g_min, self.g_max)
        return self._norm(), float(reward), False, {'secrecy': secrecy, 'P': P, 'vB': vB, 'vE': vE}

    # ---- closed-form reward gradient (used by CAPG & water-filling) -----
    def dr_dP(self, P, gB, gE):
        """d/dP [log2(1+A_B c P gB) - log2(1+A_E c P gE)] - lam   (exact)."""
        tB = self.A_B * self.c * gB
        tE = self.A_E * self.c * gE
        return (tB/(1.0 + tB*P) - tE/(1.0 + tE*P)) / LOG2 - self.lam

    def waterfill_P(self, gB, gE, iters=40):
        """Analytical optimum power per state (bisection on concave dr/dP)."""
        if self.A_B * gB <= self.A_E * gE:         # no secrecy possible at any P
            return self.P_min
        lo, hi = self.P_min, self.P_max
        if self.dr_dP(lo, gB, gE) <= 0:  return self.P_min
        if self.dr_dP(hi, gB, gE) >= 0:  return self.P_max
        for _ in range(iters):
            mid = 0.5*(lo+hi)
            if self.dr_dP(mid, gB, gE) > 0: lo = mid
            else:                            hi = mid
        return 0.5*(lo+hi)

# ---- common evaluation utility ------------------------------------------
def evaluate(policy_fn, env, n_ep=5, ep_len=200):
    """policy_fn: normalized-state -> power. Returns avg reward/step."""
    tot = 0.0
    for _ in range(n_ep):
        s = env.reset(); acc = 0.0
        for _ in range(ep_len):
            s, r, _, _ = env.step(policy_fn(s)); acc += r
        tot += acc/ep_len
    return tot/n_ep

# ---- shared plotting utility ---------------------------------------------
def plot_learning_curve(log, name, wf=None, color='#2a78d6', save_path=None):
    """
    log  : list of (step, eval_reward) tuples collected during training
    name : agent name for the title/legend
    wf   : optional water-filling baseline value (horizontal dashed line)
    Saves '<name lowercase>_curve.png' next to the script unless save_path given.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs, ys = zip(*log)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, ys, color=color, lw=2.5, marker='o', ms=5, alpha=0.9, label=name)
    if wf is not None:
        ax.axhline(wf, color='gray', ls='--', lw=1.5, alpha=0.8,
                   label=f'Water-filling ({wf:.4f})')
    ax.set_xlabel('Training steps', fontsize=13)
    ax.set_ylabel('Avg secrecy reward / step', fontsize=13)
    ax.set_title(f'{name} — Wiretap Power Control  (rate = log2(1+Av), v = b^2·P·gamma·Po)',
                 fontsize=12.5, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_path or f'{name.lower().replace(" ", "_")}_curve.png'
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[{name}] learning curve saved -> {path}')
    return path
