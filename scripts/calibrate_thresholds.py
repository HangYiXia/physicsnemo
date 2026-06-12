"""
calibrate_thresholds.py — decision gate D3 calibration (M5 step 5)
==================================================================
The AMR-Transformer threshold ranges (omega:[0.2,4.0], ...) are in the paper's
NORMALIZED scale and do not match this dataset's PHYSICAL magnitudes (|omega|~
1e2). D3 needs the real, data-driven absolute thresholds. This tool, over many
cases x frames:

    1. computes the per-L1-segment |omega| distribution (the physical scale);
    2. sweeps a set of candidate absolute omega thresholds and reports the
       resulting token-count T distribution (mean/quantiles);
    3. plots both, and recommends an omega-threshold range that keeps T in a
       healthy mid-band of [K0, K1].

It does NOT need a trained checkpoint (physical quantities depend only on the
velocity field + geometry). Use it to set K0/K1 and the train-time sampling
range for omega.

Usage:
    python calibrate_thresholds.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --stride 5
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.vortex import VortexSheddingDatasetAMR
from amr_m4gn.physics_ops import compute_ns_quantities, denormalize_velocity
from amr_m4gn.amr_router import aggregate_per_segment, route


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_cases", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--stride", type=int, default=5, help="sample every N-th frame")
    p.add_argument("--n_thresh", type=int, default=12)
    p.add_argument("--out_dir", type=str, default="./inference_vis")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = VortexSheddingDatasetAMR(
        name="amr_calib", data_dir=args.data_dir, split=args.split,
        num_samples=args.num_cases, num_steps=args.num_steps,
        noise_std=0.0, cache_dir=args.cache_dir,
    )
    vmean, vstd = ds.node_stats["velocity_mean"], ds.node_stats["velocity_std"]
    caches = {g: ds.get_cache(g) for g in range(args.num_cases)}
    K0 = caches[0]["meta"]["K0"]
    K1 = caches[0]["meta"]["K1"]
    spc = args.num_steps - 1

    # 1) collect per-segment |omega| and keep (levels, phys) per frame for routing
    frames = []          # list of (levels, phys_dict)
    all_omega_seg = []   # per-segment |omega| over all frames
    for c in range(args.num_cases):
        cache = caches[c]
        L1 = cache["levels"][1]
        K1c = int(L1.max()) + 1
        for t in range(0, spc, args.stride):
            g = ds[c * spc + t]
            u, v = denormalize_velocity(g.x[:, 0], g.x[:, 1], vmean, vstd)
            phys = compute_ns_quantities(
                u=u, v=v, pos=cache["pos"], edge_index=g.edge_index,
                area=cache["area"])
            agg = aggregate_per_segment(phys, L1, K1c)
            all_omega_seg.append(agg["omega"])
            frames.append((cache["levels"], phys))

    omega_seg = torch.cat(all_omega_seg).numpy()
    qs = [10, 30, 50, 70, 90, 95, 99]
    print("per-segment |omega| percentiles (L1, all frames):")
    for q in qs:
        print(f"  p{q:>2} = {np.percentile(omega_seg, q):.3e}")

    # 2) sweep candidate absolute omega thresholds -> T distribution
    lo, hi = np.percentile(omega_seg, 10), np.percentile(omega_seg, 95)
    cand = np.linspace(lo, hi, args.n_thresh)
    T_mean, T_lo, T_hi = [], [], []
    for thr_w in cand:
        thr = {"G": float("inf"), "omega": float(thr_w),
               "M": float("inf"), "S": float("inf")}
        Ts = [route(levels, phys, thr)[2] for (levels, phys) in frames]
        Ts = np.array(Ts)
        T_mean.append(Ts.mean()); T_lo.append(Ts.min()); T_hi.append(Ts.max())
    T_mean = np.array(T_mean)

    # 3) recommend the threshold whose mean T is closest to the K0..K1 mid-band
    target = 0.5 * (K0 + K1)
    j = int(np.argmin(np.abs(T_mean - target)))
    print(f"\nK0={K0}, K1={K1}, target mid-band T~{target:.0f}")
    print(f"recommended omega threshold ~ {cand[j]:.3e}  -> mean T = {T_mean[j]:.1f}")
    # a sampling range (train-time) that spans a useful T span:
    print(f"suggested train-time omega sampling range: "
          f"[{np.percentile(omega_seg,40):.3e}, {np.percentile(omega_seg,85):.3e}] "
          f"(p40..p85 of per-seg |omega|)")

    # plots
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    a1.hist(omega_seg, bins=60)
    a1.set_yscale("log")
    a1.set_title("per-segment |omega| distribution (L1, all frames)")
    a1.set_xlabel("|omega|"); a1.set_ylabel("count (log)")
    a2.plot(cand, T_mean, marker="o", label="mean T")
    a2.fill_between(cand, T_lo, T_hi, alpha=0.2, label="min..max T")
    a2.axhline(K0, ls="--", c="g", label=f"K0={K0}")
    a2.axhline(K1, ls="--", c="r", label=f"K1={K1}")
    a2.axvline(cand[j], ls=":", c="k", label=f"rec ~{cand[j]:.1f}")
    a2.set_title("token count T vs absolute omega threshold")
    a2.set_xlabel("omega threshold"); a2.set_ylabel("T"); a2.legend()
    fig.suptitle("D3 threshold calibration (M5)", fontsize=14)
    plt.tight_layout()
    save = os.path.join(args.out_dir, "11_threshold_calibration.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
