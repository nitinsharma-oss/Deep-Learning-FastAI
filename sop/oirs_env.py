"""
OIRS-assisted VLC environment for Problem (P1).

Physical model
--------------
LED (Lambertian order m) -> OIRS mirror element on the wall x = 0 -> photodetector.
The OIRS orientation is parameterised by two rotation angles:
    gamma : yaw   (rotation of the mirror normal about the vertical z-axis)
    omega : pitch (rotation of the mirror normal about the horizontal y-axis)
Base (unrotated) normal points into the room: n0 = (1, 0, 0).

Effective channel gain (specular point-mirror model, cf. Abdelhady et al.,
"Visible Light Communications via Intelligent Reflecting Surfaces"):

    z(gamma, omega) = rho * (m+1) * A_pd / (2*pi*(d1+d2)^2)
                      * cos^m(phi) * cos(psi) * cos^k(theta_mis) * 1{psi <= FOV}

    phi        : LED radiance angle towards the mirror
    psi        : incidence angle at the PD
    theta_mis  : angle between the specularly reflected ray and the
                 mirror -> PD direction (k large => narrow specular lobe)

ADR objective (IM/DD capacity lower bound):

    ADR = (B/2) * log2( 1 + (e / 2*pi) * ( P * R_pd * z / Q )^2 )

Analytical optimum
------------------
z is maximised when the reflected ray hits the PD exactly, i.e. when the
mirror normal is the unit bisector of the mirror->LED and mirror->PD
directions:  n* = (u_L + u_P) / ||u_L + u_P||.  gamma*, omega* follow by
inverting the rotation parameterisation.
"""

import numpy as np

# ----------------------------- system parameters -----------------------------
B      = 20e6          # modulation bandwidth [Hz]
P_tx   = 3.0           # transmitted optical power [W]
R_PD   = 0.4           # PD responsivity [A/W]
Q      = 1.0e-7        # noise standard deviation [A]
A_pd   = 1.0e-4        # PD active area [m^2]
m_lam  = 1.0           # Lambertian order (semi-angle 60 deg)
rho    = 0.95          # mirror reflectivity
k_spec = 40.0         # specular-lobe sharpness of the mirror element
FOV    = np.deg2rad(70.0)

# geometry: 5 x 5 x 3 m room
p_LED = np.array([2.5, 2.5, 3.0])   # LED on ceiling
p_IRS = np.array([0.0, 2.5, 1.5])   # OIRS element on wall x = 0
p_PD  = np.array([2.5, 1.0, 0.85])  # receiver on a table
n_LED = np.array([0.0, 0.0, -1.0])  # LED points down
# PD tilted toward the OIRS wall (standard assumption for the reflected link)
n_PD  = np.array([-1.0, 0.5, 0.8]) / np.linalg.norm([-1.0, 0.5, 0.8])

GAMMA_MIN, GAMMA_MAX = -np.pi / 2, np.pi / 2
OMEGA_MIN, OMEGA_MAX = -np.pi / 2, np.pi / 2


def _unit(v):
    return v / np.linalg.norm(v)


def mirror_normal(gamma, omega):
    """Rotate base normal n0=(1,0,0): yaw gamma about z, then pitch omega about y."""
    cg, sg = np.cos(gamma), np.sin(gamma)
    co, so = np.cos(omega), np.sin(omega)
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    Ry = np.array([[co, 0, so], [0, 1, 0], [-so, 0, co]])
    return Ry @ Rz @ np.array([1.0, 0.0, 0.0])


def channel_gain(gamma, omega):
    """Effective optical channel gain z(gamma, omega)."""
    n = mirror_normal(gamma, omega)

    v_im_led = p_LED - p_IRS            # mirror -> LED
    v_im_pd  = p_PD - p_IRS             # mirror -> PD
    d1, d2   = np.linalg.norm(v_im_led), np.linalg.norm(v_im_pd)
    u_L, u_P = v_im_led / d1, v_im_pd / d2

    # LED radiance angle towards the mirror
    cos_phi = np.dot(n_LED, -u_L)
    if cos_phi <= 0:
        return 0.0

    # incident ray direction at the mirror and specular reflection
    d_in  = -u_L                                    # LED -> mirror
    d_ref = d_in - 2.0 * np.dot(d_in, n) * n        # reflected direction

    # misalignment between reflected ray and mirror -> PD direction
    cos_mis = np.clip(np.dot(d_ref, u_P), -1.0, 1.0)
    if cos_mis <= 0:
        return 0.0

    # incidence angle at the PD
    cos_psi = np.dot(n_PD, -u_P)
    if cos_psi <= 0 or np.arccos(np.clip(cos_psi, -1, 1)) > FOV:
        return 0.0

    geom = (m_lam + 1.0) * A_pd / (2.0 * np.pi * (d1 + d2) ** 2)
    return rho * geom * (cos_phi ** m_lam) * cos_psi * (cos_mis ** k_spec)


def adr(gamma, omega):
    """Achievable data rate, eq. (1) of the problem formulation [bit/s]."""
    z = channel_gain(gamma, omega)
    snr_term = (np.e / (2.0 * np.pi)) * (P_tx * R_PD * z / Q) ** 2
    return 0.5 * B * np.log2(1.0 + snr_term)


def analytic_optimum():
    """Closed-form optimum: mirror normal = bisector of mirror->LED, mirror->PD."""
    u_L = _unit(p_LED - p_IRS)
    u_P = _unit(p_PD - p_IRS)
    n_star = _unit(u_L + u_P)
    # invert n = Ry(omega) Rz(gamma) (1,0,0):
    #   n = (cos(omega)cos(gamma), sin(gamma)... ) -- solve components:
    #   n_y = sin(gamma) is wrong under this order; derive properly:
    #   Rz(gamma)(1,0,0) = (cg, sg, 0); Ry(omega)(cg, sg, 0) = (co*cg, sg, -so*cg)
    gamma_star = np.arcsin(np.clip(n_star[1], -1, 1))
    cg = np.cos(gamma_star)
    omega_star = np.arctan2(-n_star[2] / cg, n_star[0] / cg)
    return gamma_star, omega_star, n_star


class OIRSEnv:
    """Single-step continuous-action environment for (P1).

    State  : fixed normalised geometry vector (positions of LED, OIRS, PD).
    Action : a = [gamma, omega] in [-1, 1]^2, affinely mapped to angle ranges.
    Reward : ADR(gamma, omega), normalised by a reference scale for training.
    """

    def __init__(self, reward_scale=None):
        g = np.concatenate([p_LED, p_IRS, p_PD]) / 5.0
        self.state = g.astype(np.float32)
        self.state_dim, self.action_dim = self.state.size, 2
        # reference scale so rewards are O(1)
        gs, os_, _ = analytic_optimum()
        self.adr_max_ref = adr(gs, os_)
        self.reward_scale = reward_scale or self.adr_max_ref

    def angles(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        gamma = GAMMA_MIN + (a[0] + 1.0) * 0.5 * (GAMMA_MAX - GAMMA_MIN)
        omega = OMEGA_MIN + (a[1] + 1.0) * 0.5 * (OMEGA_MAX - OMEGA_MIN)
        return gamma, omega

    def reset(self):
        return self.state.copy()

    def step(self, a):
        gamma, omega = self.angles(a)
        r_raw = adr(gamma, omega)
        return self.state.copy(), r_raw / self.reward_scale, True, {
            "adr": r_raw, "gamma": gamma, "omega": omega}


if __name__ == "__main__":
    gs, os_, ns = analytic_optimum()
    print(f"analytic  gamma* = {np.rad2deg(gs):8.4f} deg,  omega* = {np.rad2deg(os_):8.4f} deg")
    print(f"n* = {ns},  z* = {channel_gain(gs, os_):.4e},  ADR* = {adr(gs, os_)/1e6:.4f} Mbit/s")

    # brute-force verification on a fine grid
    G = np.linspace(GAMMA_MIN, GAMMA_MAX, 721)
    O = np.linspace(OMEGA_MIN, OMEGA_MAX, 721)
    best = (-1.0, None, None)
    for g in G:
        for o in O:
            v = adr(g, o)
            if v > best[0]:
                best = (v, g, o)
    print(f"grid best gamma  = {np.rad2deg(best[1]):8.4f} deg,  omega  = {np.rad2deg(best[2]):8.4f} deg,"
          f"  ADR = {best[0]/1e6:.4f} Mbit/s")
