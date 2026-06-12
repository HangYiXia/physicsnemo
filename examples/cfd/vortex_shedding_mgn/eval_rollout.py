"""
eval_rollout.py — multi-test-case rollout evaluation (M5/M6 rigor)
==================================================================
Averages the autoregressive-rollout velocity RMSE over MANY test cases (not just
case 0), for AMR-M4GN and — optionally — the MGN baseline, addressing the M5
honest-boundary "single test case" limitation. Outputs:
    - per-case step1 / final / mean RMSE table (printed + 14_eval_multicase.csv)
    - the cross-case mean RMSE-vs-step curve with a ±std band
      -> 14_eval_multicase.png

Reuses the SAME rollout as compare_baselines.py (interior nodes advanced by the
prediction, boundary kept at GT). Needs test caches for every case:
    python preprocess_partitions.py --split test --num_cases <K> --out_dir ./amr_cache

Usage:
    # AMR only, 10 test cases
    python eval_rollout.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt \
        --num_cases 10 --num_steps 90 --rollout 80 --omega_thresh 8.9
    # AMR vs MGN, 10 test cases
    python eval_rollout.py ... --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt
"""

from __future__ import annotations

import os
import csv
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.meshgraphnet import MeshGraphNet
from data_amr import make_amr_dataset
from amr_m4gn.model import AMRM4GN
from train_amr_m4gn import move_cache
from compare_baselines import rollout_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="vortex",
                   choices=["vortex", "eagle"], help="data source (M7)")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--amr_ckpt", type=str, required=True)
    p.add_argument("--mgn_ckpt", type=str, default=None,
                   help="optional: also evaluate the MGN baseline")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_cases", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=90)
    p.add_argument("--rollout", type=int, default=80)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--omega_thresh", type=float, default=8.9)
    p.add_argument("--out_dir", type=str, default="./inference_vis")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = make_amr_dataset(
        args.dataset, name="amr_eval", data_dir=args.data_dir, split=args.split,
        num_samples=args.num_cases, num_steps=args.num_steps,
        noise_std=0.0, cache_dir=args.cache_dir)
    if not ds.rollout_mask:
        raise ValueError("multi-case eval needs rollout_mask; use --split test")
    ns = {k: torch.as_tensor(v).to(device) for k, v in ds.node_stats.items()}

    amr = AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
                  processor_size=args.processor_size,
                  vel_mean=ds.node_stats["velocity_mean"],
                  vel_std=ds.node_stats["velocity_std"]).to(device)
    amr.load_state_dict(torch.load(args.amr_ckpt, map_location=device,
                                   weights_only=False)["model"])
    amr.eval()
    thr = {"G": float("inf"), "omega": args.omega_thresh,
           "M": float("inf"), "S": float("inf")}

    mgn = None
    if args.mgn_ckpt:
        mgn = MeshGraphNet(input_dim_nodes=6, input_dim_edges=3, output_dim=3,
                           processor_size=args.processor_size,
                           hidden_dim_processor=args.hidden,
                           hidden_dim_node_encoder=args.hidden,
                           hidden_dim_edge_encoder=args.hidden,
                           hidden_dim_node_decoder=args.hidden).to(device)
        mgn.load_state_dict(torch.load(args.mgn_ckpt, map_location=device,
                                       weights_only=False)["model"])
        mgn.eval()

    amr_all, mgn_all, rows = [], [], []
    for c in range(args.num_cases):
        cache = move_cache(ds.get_cache(c), device)
        mask = ds.rollout_mask[c]
        a, _ = rollout_eval(lambda g: amr(g, cache, thresholds=thr),
                            ds, c, args.rollout, ns, mask, device)
        amr_all.append(a)
        row = [c, a[0], a[-1], float(np.mean(a))]
        if mgn is not None:
            m, _ = rollout_eval(lambda g: mgn(g.x, g.edge_attr, g),
                                ds, c, args.rollout, ns, mask, device)
            mgn_all.append(m)
            row += [m[0], m[-1], float(np.mean(m))]
        rows.append(row)
        print(f"case {c:2d}: AMR mean {np.mean(a):.4e}"
              + (f" | MGN mean {np.mean(mgn_all[-1]):.4e}" if mgn is not None else ""))

    amr_arr = np.array(amr_all)            # [C, rollout]
    print(f"\n==== {args.num_cases}-case mean over rollout ({args.rollout} steps) ====")
    print(f"AMR-M4GN  step1 {amr_arr[:,0].mean():.4e}  "
          f"final {amr_arr[:,-1].mean():.4e}  mean {amr_arr.mean():.4e}")
    if mgn is not None:
        mgn_arr = np.array(mgn_all)
        print(f"MGN base  step1 {mgn_arr[:,0].mean():.4e}  "
              f"final {mgn_arr[:,-1].mean():.4e}  mean {mgn_arr.mean():.4e}")

    # CSV
    header = ["case", "amr_step1", "amr_final", "amr_mean"]
    if mgn is not None:
        header += ["mgn_step1", "mgn_final", "mgn_mean"]
    csv_path = os.path.join(args.out_dir, "14_eval_multicase.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"Saved: {csv_path}")

    # mean ± std curve across cases
    xs = range(1, args.rollout + 1)
    plt.figure(figsize=(9, 5))
    am, asd = amr_arr.mean(0), amr_arr.std(0)
    plt.plot(xs, am, label=f"AMR-M4GN (n={args.num_cases})", color="crimson")
    plt.fill_between(xs, am - asd, am + asd, alpha=0.2, color="crimson")
    if mgn is not None:
        mm, msd = mgn_arr.mean(0), mgn_arr.std(0)
        plt.plot(xs, mm, label=f"MGN baseline (n={args.num_cases})", color="steelblue")
        plt.fill_between(xs, mm - msd, mm + msd, alpha=0.2, color="steelblue")
    plt.xlabel("rollout step"); plt.ylabel("velocity RMSE (physical)")
    plt.title(f"Multi-case rollout error (mean±std over {args.num_cases} test cases)")
    plt.grid(True, alpha=0.3); plt.legend()
    save = os.path.join(args.out_dir, "14_eval_multicase.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


if __name__ == "__main__":
    main()
