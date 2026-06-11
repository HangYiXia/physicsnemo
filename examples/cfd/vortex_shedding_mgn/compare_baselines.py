"""
compare_baselines.py — AMR-M4GN vs MGN rollout comparison (M5)
==============================================================
Loads a trained AMR-M4GN checkpoint and a trained MGN-baseline checkpoint,
runs the SAME autoregressive rollout on a test case for both (interior nodes
updated by the prediction, boundary nodes kept at GT — identical to
inference.py / inference_amr_m4gn.py), and reports:
    - velocity RMSE vs rollout step (two curves) -> 12_compare_rollout.png
    - step-1 RMSE and parameter counts

Fair comparison: both models were trained with the same budget
(train_amr_m4gn_full.py / train_mgn_baseline.py, same cases/epochs/noise/NMSE).

Usage:
    python compare_baselines.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicsnemo.models.meshgraphnet import MeshGraphNet
from data_amr import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from train_amr_m4gn import move_cache


def rollout_eval(predict_fn, ds, case_idx, steps, ns, mask, device):
    """Autoregressive rollout; returns velocity RMSE per step (interior nodes).

    predict_fn(graph) -> pred[N,3] in normalized space (du, dv, p).
    Only `mask` (interior) nodes are updated by the predicted increment.
    """
    mask = mask.to(device).bool().view(-1, 1)
    mask2 = mask.repeat(1, 2)
    spc = ds.num_steps - 1
    rmse_list = []
    cur_vel_norm = None
    for t in range(steps):
        g = ds[case_idx * spc + t].to(device)
        gt_vel_t = g.x[:, 0:2] * ns["velocity_std"] + ns["velocity_mean"]
        exact_next = gt_vel_t + (g.y[:, 0:2] * ns["velocity_diff_std"] + ns["velocity_diff_mean"])
        invar = g.x.clone()
        if cur_vel_norm is not None:
            invar[:, 0:2] = cur_vel_norm
        g.x = invar
        with torch.no_grad():
            pred = predict_fn(g)
        diff = pred[:, 0:2] * ns["velocity_diff_std"] + ns["velocity_diff_mean"]
        diff = torch.where(mask2, diff, torch.zeros_like(diff))
        vt_phys = invar[:, 0:2] * ns["velocity_std"] + ns["velocity_mean"]
        new_vel = vt_phys + diff
        cur_vel_norm = (new_vel - ns["velocity_mean"]) / ns["velocity_std"]
        m = mask.view(-1)
        rmse_list.append(torch.sqrt(((new_vel[m] - exact_next[m]) ** 2).mean()).item())
    return rmse_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--amr_ckpt", type=str, required=True)
    p.add_argument("--mgn_ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--case_idx", type=int, default=0)
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

    ds = VortexSheddingDatasetAMR(
        name="amr_cmp", data_dir=args.data_dir, split=args.split,
        num_samples=args.case_idx + 1, num_steps=args.num_steps,
        noise_std=0.0, cache_dir=args.cache_dir,
    )
    ns = {k: torch.as_tensor(v).to(device) for k, v in ds.node_stats.items()}
    mask = ds.rollout_mask[args.case_idx]
    cache = move_cache(ds.get_cache(args.case_idx), device)

    # AMR-M4GN
    amr = AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
                  processor_size=args.processor_size,
                  vel_mean=ds.node_stats["velocity_mean"],
                  vel_std=ds.node_stats["velocity_std"]).to(device)
    amr.load_state_dict(torch.load(args.amr_ckpt, map_location=device,
                                   weights_only=False)["model"])
    amr.eval()
    thr = {"G": float("inf"), "omega": args.omega_thresh,
           "M": float("inf"), "S": float("inf")}

    # MGN baseline
    mgn = MeshGraphNet(input_dim_nodes=6, input_dim_edges=3, output_dim=3,
                       processor_size=args.processor_size,
                       hidden_dim_processor=args.hidden,
                       hidden_dim_node_encoder=args.hidden,
                       hidden_dim_edge_encoder=args.hidden,
                       hidden_dim_node_decoder=args.hidden).to(device)
    mgn.load_state_dict(torch.load(args.mgn_ckpt, map_location=device,
                                   weights_only=False)["model"])
    mgn.eval()

    amr_rmse = rollout_eval(lambda g: amr(g, cache, thresholds=thr),
                            ds, args.case_idx, args.rollout, ns, mask, device)
    mgn_rmse = rollout_eval(lambda g: mgn(g.x, g.edge_attr, g),
                            ds, args.case_idx, args.rollout, ns, mask, device)

    n_amr = sum(p.numel() for p in amr.parameters())
    n_mgn = sum(p.numel() for p in mgn.parameters())
    print(f"params:  AMR-M4GN {n_amr/1e6:.2f}M  |  MGN {n_mgn/1e6:.2f}M")
    print(f"step-1 velocity RMSE:  AMR {amr_rmse[0]:.3e}  |  MGN {mgn_rmse[0]:.3e}")
    print(f"final   velocity RMSE:  AMR {amr_rmse[-1]:.3e}  |  MGN {mgn_rmse[-1]:.3e}")
    print(f"mean    velocity RMSE:  AMR {np.mean(amr_rmse):.3e}  |  MGN {np.mean(mgn_rmse):.3e}")

    plt.figure(figsize=(9, 5))
    xs = range(1, args.rollout + 1)
    plt.plot(xs, amr_rmse, marker=".", label=f"AMR-M4GN ({n_amr/1e6:.2f}M)")
    plt.plot(xs, mgn_rmse, marker=".", label=f"MGN baseline ({n_mgn/1e6:.2f}M)")
    plt.xlabel("rollout step"); plt.ylabel("velocity RMSE (physical)")
    plt.title(f"Rollout error: AMR-M4GN vs MGN (case {args.case_idx})")
    plt.grid(True, alpha=0.3); plt.legend()
    save = os.path.join(args.out_dir, f"12_compare_rollout_case{args.case_idx}.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


if __name__ == "__main__":
    main()
