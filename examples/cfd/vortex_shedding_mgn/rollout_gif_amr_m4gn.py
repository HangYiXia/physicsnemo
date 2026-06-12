"""
rollout_gif_amr_m4gn.py — animated field GIF for AMR-M4GN (M5)
=============================================================
The AMR-M4GN counterpart of the stock `inference.py` GIF: runs an autoregressive
rollout of a trained AMR-M4GN checkpoint on a test case and saves a 2-row
animation (top: prediction, bottom: ground-truth) of the chosen field over the
rollout, exactly like inference.py's `animation_<var>.gif`.

Rollout is identical to compare_baselines.py / inference_amr_m4gn.py: only the
interior (`rollout_mask`) nodes are advanced by the predicted velocity
increment; boundary nodes keep GT. Pressure (`p`) is the model's absolute output.

Usage:
    python rollout_gif_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt \
        --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9 \
        --fields u v p
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import tri as mtri
from matplotlib.patches import Rectangle

from data_amr import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from train_amr_m4gn import move_cache


VAR_ID = {"u": 0, "v": 1, "p": 2}


def run_rollout(model, ds, cache, case_idx, steps, ns, thr, device):
    """Autoregressive rollout; returns (pred_phys, exact_phys) lists of [N,3]
    physical-unit fields (u, v, p) per step."""
    mask = ds.rollout_mask[case_idx].to(device).bool().view(-1, 1)
    mask2 = mask.repeat(1, 2)
    spc = ds.num_steps - 1
    preds, exacts = [], []
    cur_vel_norm = None
    for t in range(steps):
        g = ds[case_idx * spc + t].to(device)
        gt_vel_t = g.x[:, 0:2] * ns["velocity_std"] + ns["velocity_mean"]
        exact_vel = gt_vel_t + (g.y[:, 0:2] * ns["velocity_diff_std"] + ns["velocity_diff_mean"])
        exact_p = g.y[:, 2] * ns["pressure_std"] + ns["pressure_mean"]

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
        pred_p = pred[:, 2] * ns["pressure_std"] + ns["pressure_mean"]

        preds.append(torch.cat([new_vel, pred_p.unsqueeze(1)], dim=1).cpu().numpy())
        exacts.append(torch.cat([exact_vel, exact_p.unsqueeze(1)], dim=1).cpu().numpy())
    return preds, exacts


def make_gif(pos, faces, preds, exacts, var_idx, var_name, save_path, frame_skip):
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    pred_i = [p[:, var_idx] for p in preds]
    exact_i = [e[:, var_idx] for e in exacts]

    plt.rcParams["image.cmap"] = "inferno"
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    fig.set_facecolor("black")
    ax[0].set_facecolor("black")
    ax[1].set_facecolor("black")

    # fixed color scale across frames (GT range) so the animation is comparable
    vmin = float(np.min([e.min() for e in exact_i]))
    vmax = float(np.max([e.max() for e in exact_i]))

    def animate(num):
        n = num * frame_skip
        for a, field, title in (
            (ax[0], pred_i[n], f"AMR-M4GN Prediction ({var_name})"),
            (ax[1], exact_i[n], f"Ground Truth ({var_name})"),
        ):
            a.cla()
            a.set_axis_off()
            a.add_patch(Rectangle((0, 0), 1.4, 0.4, facecolor="navy"))
            a.tripcolor(triang, field, vmin=vmin, vmax=vmax)
            a.triplot(triang, "ko-", ms=0.5, lw=0.3)
            a.set_title(title, color="white")
            a.set_aspect("auto", adjustable="box")
            a.autoscale(enable=True, tight=True)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0.1, hspace=0.2)
        return fig

    ani = animation.FuncAnimation(
        fig, animate, frames=len(pred_i) // frame_skip, interval=100)
    ani.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--case_idx", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=90)
    p.add_argument("--rollout", type=int, default=80)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--omega_thresh", type=float, default=8.9)
    p.add_argument("--fields", type=str, nargs="+", default=["u", "v", "p"],
                   choices=list(VAR_ID.keys()))
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./animations")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = VortexSheddingDatasetAMR(
        name="amr_gif", data_dir=args.data_dir, split=args.split,
        num_samples=args.case_idx + 1, num_steps=args.num_steps,
        noise_std=0.0, cache_dir=args.cache_dir,
    )
    if not ds.rollout_mask:
        raise ValueError("rollout GIF needs rollout_mask; use --split test")
    ns = {k: torch.as_tensor(v).to(device) for k, v in ds.node_stats.items()}
    cache = move_cache(ds.get_cache(args.case_idx), device)

    model = AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
                    processor_size=args.processor_size,
                    vel_mean=ds.node_stats["velocity_mean"],
                    vel_std=ds.node_stats["velocity_std"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device,
                                     weights_only=False)["model"])
    model.eval()
    thr = {"G": float("inf"), "omega": args.omega_thresh,
           "M": float("inf"), "S": float("inf")}

    preds, exacts = run_rollout(model, ds, cache, args.case_idx,
                                args.rollout, ns, thr, device)
    pos = cache["pos"].cpu().numpy()
    faces = np.asarray(ds.cells[args.case_idx])

    for f in args.fields:
        save = os.path.join(args.out_dir,
                            f"amr_m4gn_case{args.case_idx}_{f}.gif")
        make_gif(pos, faces, preds, exacts, VAR_ID[f], f, save, args.frame_skip)


if __name__ == "__main__":
    main()
