# Solution of Problem (P1): Optimal OIRS Orientation via DRL

## 1. Analytical solution (ground truth)

The ADR is monotonically increasing in the channel gain z(γ,ω), and z is
maximized when the specularly reflected ray hits the PD exactly, i.e. when the
mirror normal equals the **unit bisector** of the mirror→LED and mirror→PD
directions:

    n* = (u_L + u_P) / ||u_L + u_P||,
    u_L = (p_LED − p_IRS)/||·||,  u_P = (p_PD − p_IRS)/||·||

With the normal parameterisation n(γ,ω) = Ry(ω)Rz(γ)·(1,0,0)ᵀ
= (cos ω cos γ, sin γ, −sin ω cos γ), inverting gives

    γ* = arcsin(n*_y),   ω* = arctan2(−n*_z / cos γ*, n*_x / cos γ*).

For the simulated geometry (5×5×3 m room; LED (2.5, 2.5, 3.0); OIRS (0, 2.5, 1.5);
PD (2.5, 1.0, 0.85), tilted toward the OIRS wall):

    γ* = −16.2733°,  ω* = −9.9382°,  ADR* = 35.199 Mbit/s

Verified by exhaustive 721×721 grid search (best on-grid point −16.25°, −10.00°,
ADR 35.196 Mbit/s — below ADR* only by grid resolution).

## 2. DRL solution of (P1)

Single-step episodes: agent outputs a = [γ, ω] ∈ [−1,1]² (affinely mapped to
[−π/2, π/2]²); the environment returns ADR(γ,ω) as reward. Identical state,
action space, reward, network width (64), and budget (4,000 steps) for all
five algorithms.

| Algorithm | γ (deg) | ω (deg) | ADR (Mbit/s) | % of optimum |
|-----------|---------|---------|--------------|--------------|
| DDPG      | −16.283 | −10.024 | 35.195       | 99.99 %      |
| SAC       | −16.474 |  −9.950 | 35.176       | 99.94 %      |
| ECDPO*    | −16.426 | −10.290 | 35.124       | 99.79 %      |
| CAPG      | −18.116 |  −9.883 | 33.298       | 94.60 %      |
| PPO       |  −9.679 | −10.096 | 13.855       | 39.36 %      |
| **Analytic optimum** | **−16.273** | **−9.938** | **35.199** | 100 % |

\* ECDPO implemented here as a placeholder (DDPG backbone + twin critics +
delayed policy updates + decaying exploration noise). Substitute the actual
ECDPO update rule from the paper before reporting results.

## 3. Findings

- Off-policy actor–critic methods (DDPG, SAC, ECDPO) recover the analytical
  optimum to within 0.2 % in ≤ 4,000 interactions.
- CAPG (REINFORCE-style) converges close to the optimum but with higher
  variance in the learned angles.
- PPO is sample-starved at this budget (on-policy), reaching ~39 % of the
  optimum — increase the interaction budget or batch size for a fair on-policy
  comparison, or report this sample-efficiency gap as a finding.
- With an idealised, very narrow specular lobe (k = 300) the reward becomes
  near-sparse and only DDPG/ECDPO succeed; a realistic lobe width (k = 40) was
  used for the reported comparison. State the lobe model explicitly in the paper.

## 4. Files

- `oirs_env.py`  — channel model, ADR objective, analytical optimum, grid check
- `train_drl.py` — DDPG, SAC, PPO, CAPG, ECDPO implementations and training
- `results.json` — learned angles and ADR per algorithm
- `convergence.png` — convergence curves vs. analytical optimum
