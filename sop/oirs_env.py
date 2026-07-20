"""
oirs_vlc_env.py
===============
Research-aligned OIRS-VLC environment (Sharma, Mathur, Bhandari framework).

Implements the system of the paper "Cross-Layer Reliability Optimization of
OIRS-Aided VLC Systems under Random UE Orientation":

  * Complete LoS blockage: only the OIRS-assisted NLoS path exists.
  * OIRS with N elements acting as a unified reflector with COMMON yaw gamma
    and roll omega, serving all K users simultaneously (paper Sec. 2).
  * Aggregate NLoS channel gain, paper Eq. (1):
        Z_k = sum_{l=1..M} sum_{i=1..N} X_{k,i,l},   X = a_k * cos(psi_k^i)
  * Random UE orientation: xi = cos(psi) ~ truncated Laplace on
    [xi_min, xi_max] with parameters (mu_xi, b_xi) -- paper Eq. (4),
    worked-example values mu_xi = 0.8789, b_xi = 0.036, xi in [0, 0.9574].
  * Exact moments E[u], E[u^2] -- paper Eqs. (6)-(8).
  * Gamma moment matching  k_g = MN E[xi]^2/Var[xi], th_g = a_k Var[xi]/E[xi]
    -- paper Eq. (13).
  * IM/DD OOK capacity bound: ADR_k = (B/2) log2(1 + (e/2pi)(P R_pd Z_k / Q)^2).
  * HARQ-IR outage (Gamma CDF <=> Meijer-G reduction) and average AoI,
    paper Eqs. (17)-(25), for cross-layer evaluation.

RL problem (aligned with the paper, unlike the earlier deterministic model):
  state  : fixed normalised geometry of AP / OIRS / users
  action : a = [gamma, omega] in [-1,1]^2 -> [-pi/2, pi/2]^2   (constraint 26a)
  reward : STOCHASTIC -- one fresh draw of all MN orientation cosines per user
           per step; reward = mean_k ADR_k / scale.  The agent must therefore
           maximise the EXPECTED ADR under random UE orientation; no
           deterministic closed-form optimum exists.
"""

import numpy as np

try:
    from scipy.special import gammainc  # regularised lower incomplete gamma
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# --------------------------- ADR / noise parameters --------------------------
B     = 20e6      # bandwidth [Hz]
P_tx  = 3.0       # transmitted optical power [W]
R_PD  = 0.4       # PD responsivity [A/W]
Q_n   = 1.0e-7    # noise standard deviation [A]

# ------------------- truncated-Laplace orientation statistics ----------------
# worked-example values from the paper / MATLAB verification script
B_XI     = 0.03600
XI_MIN   = 0.0        # FoV limit (paper: lower bound from receiver FoV)
XI_MAX   = 0.9574     # upper geometric bound

# ------------------------------- OIRS optics ---------------------------------
RHO     = 0.95        # reflectivity
A_PD    = 1.0e-4      # PD area [m^2]
M_LAM   = 1.0         # Lambertian order
K_SPEC  = 20.0        # unified-reflector beam concentration exponent

GAMMA_MIN, GAMMA_MAX = -np.pi / 2, np.pi / 2   # constraint (26a)
OMEGA_MIN, OMEGA_MAX = -np.pi / 2, np.pi / 2


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# =========================== truncated-Laplace tools ==========================
def trunc_laplace_moments(mu, b=B_XI, lo=XI_MIN, hi=XI_MAX):
    """Exact moments of xi ~ truncated Laplace -- paper Eqs. (5)-(8)."""
    a_ = lo - mu   # alpha <= 0
    be = hi - mu   # beta  >= 0
    lam = b * (2.0 - np.exp(a_ / b) - np.exp(-be / b))                # Eq. (5)
    e_u = (b / lam) * (np.exp(a_ / b) * (b - a_)
                       - np.exp(-be / b) * (be + b))                  # Eq. (6)
    e_u2 = (1.0 / lam) * (4 * b ** 3
                          - np.exp(a_ / b) * (b * a_ ** 2 - 2 * b ** 2 * a_ + 2 * b ** 3)
                          - np.exp(-be / b) * (b * be ** 2 + 2 * b ** 2 * be + 2 * b ** 3))  # Eq. (8)
    e_xi = mu + e_u
    var_xi = e_u2 - e_u ** 2
    return e_xi, var_xi


def sample_trunc_laplace(mu, size, rng, b=B_XI, lo=XI_MIN, hi=XI_MAX):
    """Accept-reject sampler -- mirrors the MATLAB verification script."""
    shape = (size,) if np.isscalar(size) else tuple(size)
    size = int(np.prod(shape))
    out = np.empty(size, dtype=np.float64)
    filled = 0
    while filled < size:
        n = 2 * (size - filled) + 16
        u = rng.random(n) - 0.5
        s = mu - b * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))
        s = s[(s >= lo) & (s <= hi)]
        take = min(s.size, size - filled)
        out[filled:filled + take] = s[:take]
        filled += take
    return out.reshape(shape)


# ================================ environment =================================
class OIRSVLCEnv:
    """Multi-user OIRS-VLC environment with random UE orientation.

    Single-step episodes: a = [gamma, omega] -> stochastic reward mean_k ADR_k.
    """

    def __init__(self, K=3, M=1, N=4, seed=0, n_ref=1500, reward_mode='adr',
                 rate_R=2.0, gamma_bar=1e14, L_harq=4, k_spec=K_SPEC):
        self.k_spec = float(k_spec)
        assert reward_mode in ('adr', 'aoi')
        self.reward_mode = reward_mode
        self.rate_R, self.gamma_bar, self.L_harq = rate_R, gamma_bar, L_harq
        self.rng = np.random.default_rng(seed)
        self.M, self.N, self.K = M, N, K
        self.MN = M * N

        # ------------------------- geometry (5 x 5 x 3 m) --------------------
        self.p_led = np.array([2.5, 2.5, 3.0])
        self.p_irs = np.array([0.0, 2.5, 1.5])
        self.n_led = np.array([0.0, 0.0, -1.0])
        self.users = np.array([[2.0, 1.0, 0.85],
                               [3.0, 3.5, 0.85],
                               [1.5, 4.0, 0.85]])[:K]

        # LED -> IRS leg (independent of the OIRS orientation)
        v = self.p_led - self.p_irs
        self.d1 = np.linalg.norm(v)
        self.u_led = v / self.d1
        self.cos_phi = max(0.0, float(np.dot(self.n_led, -self.u_led)))

        # IRS -> user legs and mean incidence cosines mu_xi_k
        self.d2 = np.zeros(K)
        self.u_usr = np.zeros((K, 3))
        self.mu_xi = np.zeros(K)
        for k in range(K):
            w = self.users[k] - self.p_irs
            self.d2[k] = np.linalg.norm(w)
            self.u_usr[k] = w / self.d2[k]
            # mean UE normal: tilted between "up" and "towards the OIRS wall"
            n_bar = _unit(_unit(self.p_irs - self.users[k]) + np.array([0, 0, 1.2]))
            self.mu_xi[k] = float(np.clip(np.dot(n_bar, -self.u_usr[k]),
                                          0.20, 0.95 * XI_MAX))

        # exact per-user moments (paper Eqs. 6-8) and Gamma shapes (Eq. 13)
        self.e_xi = np.zeros(K)
        self.var_xi = np.zeros(K)
        for k in range(K):
            self.e_xi[k], self.var_xi[k] = trunc_laplace_moments(self.mu_xi[k])
        self.k_g = self.MN * self.e_xi ** 2 / self.var_xi      # shape (a_k-free)

        # common-random-number reference sums S_k = sum_{MN} xi  (CRN evaluation)
        self.S_ref = np.stack([
            sample_trunc_laplace(self.mu_xi[k], (n_ref, self.MN), self.rng).sum(axis=1)
            for k in range(K)])                                 # (K, n_ref)

        # RL interface
        g = np.concatenate([self.p_led, self.p_irs, self.users.ravel()]) / 5.0
        self.state = g.astype(np.float32)
        self.state_dim, self.action_dim = self.state.size, 2

        # reward scale from a coarse grid of the expected ADR
        self.reward_scale = max(self._coarse_best()[2], 1e-6)

    # ---------------------------- channel geometry ---------------------------
    def mirror_normal(self, gamma, omega):
        cg, sg = np.cos(gamma), np.sin(gamma)
        co, so = np.cos(omega), np.sin(omega)
        return np.array([co * cg, sg, -so * cg])

    def a_k(self, gamma, omega):
        """Per-user aggregated path gain a_k(gamma, omega) (paper Eq. 1 text):
        Lambertian LED->IRS emission, unified-reflector beam steering toward
        each user, spherical spreading over d1 + d2k."""
        n = self.mirror_normal(gamma, omega)
        d_in = -self.u_led
        d_ref = d_in - 2.0 * np.dot(d_in, n) * n
        c_mis = np.clip(self.u_usr @ d_ref, 0.0, 1.0)           # (K,)
        geom = (M_LAM + 1.0) * A_PD / (2.0 * np.pi * (self.d1 + self.d2) ** 2)
        return RHO * geom * (self.cos_phi ** M_LAM) * (c_mis ** self.k_spec)

    # ------------------------------ ADR / reward -----------------------------
    @staticmethod
    def adr_of_Z(Z):
        snr = (np.e / (2.0 * np.pi)) * (P_tx * R_PD * Z / Q_n) ** 2
        return 0.5 * B * np.log2(1.0 + snr)

    def sample_adr(self, gamma, omega):
        """One stochastic draw of all MN orientation cosines per user."""
        ak = self.a_k(gamma, omega)
        S = np.array([sample_trunc_laplace(self.mu_xi[k], self.MN, self.rng).sum()
                      for k in range(self.K)])
        return self.adr_of_Z(ak * S)                            # (K,)

    def expected_adr(self, gamma, omega):
        """E[ADR_k] over UE orientation via common random numbers."""
        ak = self.a_k(gamma, omega)                             # (K,)
        Z = ak[:, None] * self.S_ref                            # (K, n_ref)
        return self.adr_of_Z(Z).mean(axis=1)                    # (K,)

    def _coarse_best(self, ng=61):
        gs = np.linspace(GAMMA_MIN, GAMMA_MAX, ng)
        os_ = np.linspace(OMEGA_MIN, OMEGA_MAX, ng)
        best = (0.0, 0.0, -1.0)
        for g in gs:
            for o in os_:
                v = self.expected_adr(g, o).mean()
                if v > best[2]:
                    best = (g, o, v)
        return best

    def grid_reference(self, ng=181, refine=41, span=0.08):
        """Fine grid + local refinement of the expected mean ADR (benchmark)."""
        g0, o0, _ = self._coarse_best(ng=ng)
        gs = np.linspace(g0 - span, g0 + span, refine)
        os_ = np.linspace(o0 - span, o0 + span, refine)
        best = (g0, o0, self.expected_adr(g0, o0).mean())
        for g in gs:
            for o in os_:
                v = self.expected_adr(g, o).mean()
                if v > best[2]:
                    best = (g, o, v)
        return best

    # ------------------------------ RL interface -----------------------------
    def angles(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        gamma = GAMMA_MIN + (a[0] + 1.0) * 0.5 * (GAMMA_MAX - GAMMA_MIN)
        omega = OMEGA_MIN + (a[1] + 1.0) * 0.5 * (OMEGA_MAX - OMEGA_MIN)
        return gamma, omega

    def reset(self):
        return self.state.copy()

    def step(self, a):
        gamma, omega = self.angles(a)
        adr_k = self.sample_adr(gamma, omega)                   # stochastic!
        aoi_max = float(self.avg_aoi_ir(gamma, omega, self.rate_R,
                                        self.gamma_bar, self.L_harq).max())
        if self.reward_mode == 'adr':
            r = adr_k.mean() / self.reward_scale                # maximise E[ADR]
        else:  # 'aoi' -- paper objective (26): min-max AoI, deterministic
            r = -aoi_max / (self.L_harq + 0.5)
        return self.state.copy(), float(r), True, {
            "adr_mean": float(adr_k.mean()), "adr_k": adr_k,
            "aoi_max": aoi_max, "gamma": gamma, "omega": omega}

    # -------------------- cross-layer evaluation (HARQ-IR AoI) ---------------
    def outage_ir(self, gamma, omega, l, rate_R, gamma_bar):
        """P_out,l^IR via the Gamma CDF (the paper's Meijer G_{1,2}^{1,1}
        reduces to the regularised incomplete gamma function)."""
        if not _HAVE_SCIPY:
            raise RuntimeError("scipy required for outage/AoI evaluation")
        ak = self.a_k(gamma, omega)
        th_g = ak * self.var_xi / self.e_xi                     # Eq. (13)
        U = np.sqrt((2 * np.pi / np.e) * (2.0 ** (2 * rate_R / l) - 1.0) / gamma_bar)
        LB = self.MN * ak * XI_MIN                              # physical bound
        th_safe = np.maximum(th_g, 1e-300)                      # a_k -> 0 guard
        p = gammainc(self.k_g, U / th_safe) \
            - gammainc(self.k_g, np.maximum(LB, 0.0) / th_safe)
        p = np.where(ak <= 0.0, 1.0, p)                         # no link => outage
        return np.clip(p, 0.0, 1.0)                             # (K,)

    def avg_aoi_ir(self, gamma, omega, rate_R=None, gamma_bar=None, L=None):
        rate_R = self.rate_R if rate_R is None else rate_R
        gamma_bar = self.gamma_bar if gamma_bar is None else gamma_bar
        L = self.L_harq if L is None else L
        """Average AoI per user, paper Eqs. (24)-(25); returns (K,) array."""
        ET = np.ones(self.K)
        ET2 = np.ones(self.K)
        for l in range(1, L):
            p = self.outage_ir(gamma, omega, l, rate_R, gamma_bar)
            ET += p
            ET2 += (2 * l + 1) * p
        return ET + (ET2 - ET ** 2) / (2 * ET) + 0.5


if __name__ == "__main__":
    env = OIRSVLCEnv(seed=0)
    print(f"mu_xi per user      : {np.round(env.mu_xi, 4)}")
    print(f"E[xi] per user      : {np.round(env.e_xi, 4)}")
    print(f"Var[xi] per user    : {env.var_xi}")
    print(f"Gamma shape k_g     : {np.round(env.k_g, 2)}")
    g, o, v = env.grid_reference()
    print(f"grid reference      : gamma = {np.rad2deg(g):.3f} deg, "
          f"omega = {np.rad2deg(o):.3f} deg, E[mean ADR] = {v/1e6:.3f} Mbit/s")
    print(f"per-user E[ADR]     : {np.round(env.expected_adr(g, o)/1e6, 3)} Mbit/s")
    print(f"per-user avg AoI    : {np.round(env.avg_aoi_ir(g, o), 3)} rounds")
