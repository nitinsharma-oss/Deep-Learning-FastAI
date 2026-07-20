"""train_all.py -- trains DDPG, SAC, PPO, CAPG, ECDPO on the research-aligned
OIRS-VLC environment (random UE orientation, K users, stochastic ADR reward)
and evaluates every learned orientation on:

  * expected mean ADR  E[(1/K) sum_k ADR_k]  (common random numbers),
  * per-user expected ADR,
  * per-user average HARQ-IR AoI (paper Eqs. 24-25),

against a fine grid-search reference of the expected-ADR optimum.
Bonus run: ECDPO retrained with the paper's min-max AoI objective (Eq. 26)
to demonstrate the fairness/mean-throughput trade-off.
"""
import json
import time

import numpy as np

from oirs_vlc_env import OIRSVLCEnv
import ddpg, sac, ppo, capg, ecdpo

TOTAL_STEPS = 6000
SEED = 0


def evaluate(env, a_fin):
    gamma, omega = env.angles(a_fin)
    e_adr_k = env.expected_adr(gamma, omega)
    aoi_k = env.avg_aoi_ir(gamma, omega)
    return dict(gamma_deg=float(np.rad2deg(gamma)),
                omega_deg=float(np.rad2deg(omega)),
                mean_adr_mbps=float(e_adr_k.mean() / 1e6),
                adr_k_mbps=[float(x) for x in e_adr_k / 1e6],
                aoi_k=[float(x) for x in aoi_k],
                aoi_max=float(aoi_k.max()))


if __name__ == "__main__":
    results, curves = {}, {}

    for mod in (ddpg, sac, ppo, capg, ecdpo):
        env = OIRSVLCEnv(seed=SEED)               # fresh env, same seed: fair
        t0 = time.time()
        name, curve, a_fin = mod.train(env, total_steps=TOTAL_STEPS, seed=SEED)
        res = evaluate(env, a_fin)
        res["train_s"] = round(time.time() - t0, 1)
        results[name] = res
        curves[name] = curve
        print(f"{name:6s} g={res['gamma_deg']:8.3f}  w={res['omega_deg']:8.3f}  "
              f"E[ADR]={res['mean_adr_mbps']:7.3f} Mb/s  "
              f"AoImax={res['aoi_max']:.2f}  ({res['train_s']}s)")

    # grid-search reference of the expected-ADR optimum
    env = OIRSVLCEnv(seed=SEED)
    g, o, v = env.grid_reference()
    ref = dict(gamma_deg=float(np.rad2deg(g)), omega_deg=float(np.rad2deg(o)),
               mean_adr_mbps=float(v / 1e6),
               adr_k_mbps=[float(x) for x in env.expected_adr(g, o) / 1e6],
               aoi_k=[float(x) for x in env.avg_aoi_ir(g, o)])
    results["Grid reference"] = ref
    print(f"{'GRID':6s} g={ref['gamma_deg']:8.3f}  w={ref['omega_deg']:8.3f}  "
          f"E[ADR]={ref['mean_adr_mbps']:7.3f} Mb/s")

    # bonus: paper objective (26) -- min-max AoI with the best agent
    env_aoi = OIRSVLCEnv(seed=SEED, reward_mode="aoi", gamma_bar=3e15,
                     k_spec=6.0)  # fairness config: wide beam
    name, curve, a_fin = ecdpo.train(env_aoi, total_steps=TOTAL_STEPS,
                                     seed=SEED, label="ECDPO-AoI")
    res = evaluate(env_aoi, a_fin)
    results["ECDPO-AoI"] = res
    curves["ECDPO-AoI"] = curve
    print(f"{'E-AoI':6s} g={res['gamma_deg']:8.3f}  w={res['omega_deg']:8.3f}  "
          f"E[ADR]={res['mean_adr_mbps']:7.3f} Mb/s  AoImax={res['aoi_max']:.2f}")

    np.savez("curves.npz", **curves)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
