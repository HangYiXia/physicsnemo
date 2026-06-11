"""
inference_amr_m4gn.py — single-step prediction + visualization (M5 step 4)
==========================================================================
Loads a trained AMRM4GN checkpoint, predicts one frame of a case, denormalizes,
and plots prediction vs ground-truth vs |error| for (du, dv, p) as a 3x3 panel
(09_prediction.png). Also prints per-channel normalized error.

This answers "can I finally SEE the result?": the panel shows how close the
model's predicted increment/pressure fields are to the truth, drawn on the mesh.
(Multi-step rollout + full Sec-6.4 metrics + baseline comparison come next.)

Usage:
    python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch49.pt \
        --case_idx 0 --timestep 25 --num_steps 50 --omega_thresh 30
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from data_amr import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from train_amr_m4gn import move_cache


def _denorm(x, mean, std):
    return x * torch.as_tensor(std, dtype=x.dtype, device=x.device) \
             + torch.as_tensor(mean, dtype=x.dtype, device=x.device)


def plot_prediction(pos, cells, pred, true, save_path):
    """pred/true: [N,3] = (du, dv, p) in PHYSICAL units. 3x3 panel."""
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    names = ["du", "dv", "p"]
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    for r in range(3):
        err = np.abs(pred[:, r] - true[:, r])
        fields = [pred[:, r], true[:, r], err]
        titles = [f"{names[r]} predicted", f"{names[r]} ground-truth",
                  f"{names[r]} |error|"]
        # symmetric scale for the field columns, sequential for error
        vmax = np.percentile(np.abs(true[:, r]), 99) + 1e-12
        for c in range(3):
            ax = axes[r, c]
            if c < 2:
                tc = ax.tripcolor(triang, fields[c], cmap="RdBu_r",
                                  shading="gouraud", vmin=-vmax, vmax=vmax)
            else:
                tc = ax.tripcolor(triang, fields[c], cmap="viridis",
                                  shading="gouraud", vmin=0,
                                  vmax=np.percentile(err, 99) + 1e-12)
            ax.set_aspect("equal")
            ax.set_title(titles[c])
            fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("AMR-M4GN prediction vs ground-truth (M5)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_rollout(model, ds, cache, case_idx, steps, thr, device, out_dir):
    """Autoregressive rollout (baseline-style): only `rollout_mask` (interior)
    nodes are updated by the predicted increment; boundary nodes keep GT. Each
    step feeds the previous predicted velocity as the next input. Plots velocity
    RMSE vs rollout step (error accumulation).
    """
    ns = {k: torch.as_tensor(v).to(device) for k, v in ds.node_stats.items()}
    mask = ds.rollout_mask[case_idx].to(device).bool().view(-1, 1)  # [N,1], True=update
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
            pred = model(g, cache, thresholds=thr)
        diff = pred[:, 0:2] * ns["velocity_diff_std"] + ns["velocity_diff_mean"]
        diff = torch.where(mask2, diff, torch.zeros_like(diff))
        vt_phys = invar[:, 0:2] * ns["velocity_std"] + ns["velocity_mean"]
        new_vel = vt_phys + diff
        cur_vel_norm = (new_vel - ns["velocity_mean"]) / ns["velocity_std"]
        m = mask.view(-1)
        rmse = torch.sqrt(((new_vel[m] - exact_next[m]) ** 2).mean()).item()
        rmse_list.append(rmse)

    print("  rollout velocity RMSE per step (first/mid/last): "
          f"{rmse_list[0]:.3e} / {rmse_list[len(rmse_list)//2]:.3e} / {rmse_list[-1]:.3e}")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, steps + 1), rmse_list, marker=".")
    plt.xlabel("rollout step"); plt.ylabel("velocity RMSE (physical)")
    plt.title(f"AMR-M4GN rollout error accumulation (case {case_idx})")
    plt.grid(True, alpha=0.3)
    save = os.path.join(out_dir, f"10_rollout_rmse_case{case_idx}.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save}")
    return rmse_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--case_idx", type=int, default=0)
    p.add_argument("--timestep", type=int, default=25)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--omega_thresh", type=float, default=30.0)
    p.add_argument("--rollout", type=int, default=0,
                   help="if >0, autoregressive rollout this many steps; needs "
                        "rollout_mask -> use --split test")
    p.add_argument("--out_dir", type=str, default="./inference_vis")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = VortexSheddingDatasetAMR(
        name="amr_infer", data_dir=args.data_dir, split=args.split,
        num_samples=args.case_idx + 1, num_steps=args.num_steps,
        noise_std=0.0, cache_dir=args.cache_dir,
    )
    cache = move_cache(ds.get_cache(args.case_idx), device)
    ns = ds.node_stats

    model = AMRM4GN(
        in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
        processor_size=args.processor_size,
        vel_mean=ns["velocity_mean"], vel_std=ns["velocity_std"],
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    thr = {"G": float("inf"), "omega": args.omega_thresh,
           "M": float("inf"), "S": float("inf")}

    if args.rollout > 0:
        if not ds.rollout_mask:
            raise ValueError("rollout needs rollout_mask; use --split test")
        run_rollout(model, ds, cache, args.case_idx, args.rollout, thr,
                    device, args.out_dir)
        print("done.")
        return

    idx = args.case_idx * (args.num_steps - 1) + args.timestep
    graph = ds[idx].to(device)
    with torch.no_grad():
        pred = model(graph, cache, thresholds=thr)   # [N,3] normalized space
    true = graph.y

    # normalized-space per-channel NMSE (same as training loss scale)
    nmse = ((pred - true) ** 2).mean(0) / ((true ** 2).mean(0) + 1e-8)
    print(f"  per-channel NMSE (norm space) du/dv/p: "
          f"{nmse[0]:.3e} / {nmse[1]:.3e} / {nmse[2]:.3e}")

    # denormalize to physical units for plotting
    pred_phys = torch.empty_like(pred)
    true_phys = torch.empty_like(true)
    pred_phys[:, :2] = _denorm(pred[:, :2], ns["velocity_diff_mean"], ns["velocity_diff_std"])
    true_phys[:, :2] = _denorm(true[:, :2], ns["velocity_diff_mean"], ns["velocity_diff_std"])
    pred_phys[:, 2] = _denorm(pred[:, 2], ns["pressure_mean"], ns["pressure_std"])
    true_phys[:, 2] = _denorm(true[:, 2], ns["pressure_mean"], ns["pressure_std"])

    rmse = torch.sqrt(((pred_phys - true_phys) ** 2).mean(0))
    print(f"  per-channel RMSE (physical)   du/dv/p: "
          f"{rmse[0]:.3e} / {rmse[1]:.3e} / {rmse[2]:.3e}")

    pos = cache["pos"].cpu().numpy()
    cells = np.asarray(ds.cells[args.case_idx])
    plot_prediction(
        pos, cells, pred_phys.cpu().numpy(), true_phys.cpu().numpy(),
        os.path.join(args.out_dir, f"09_prediction_case{args.case_idx}_t{args.timestep}.png"))
    print("done.")


if __name__ == "__main__":
    main()
