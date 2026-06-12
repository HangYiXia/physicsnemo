"""
inference_amr_m4gn.py — prediction + visualization (single-step / rollout / GIF) (M5)
====================================================================================
Loads a trained AMRM4GN checkpoint and visualizes it in three modes:

  (default)    single-step: predict one frame, plot pred / GT / |error| for
               (du, dv, p) as a 3x3 panel (09_prediction.png) + per-channel
               NMSE/RMSE.
  --rollout R  autoregressive rollout of R steps; plot velocity RMSE vs step
               (10_rollout_rmse.png), the error-accumulation curve.
  --gif        on top of a rollout, save an animated field GIF (top: prediction,
               bottom: ground-truth) for each --gif_fields var, in the same
               style as the stock inference.py (animations/amr_m4gn_*.gif).

`--rollout`/`--gif` need the interior `rollout_mask`, so use `--split test`.

Usage:
    # single-step panel
    python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt \
        --case_idx 0 --timestep 25 --num_steps 50 --omega_thresh 8.9
    # rollout RMSE curve + field GIFs (u/v/p)
    python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt \
        --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9 \
        --gif --gif_fields u v p
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
from matplotlib import animation
from matplotlib.patches import Rectangle

from data_amr import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from train_amr_m4gn import move_cache


VAR_ID = {"u": 0, "v": 1, "p": 2}


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
    RMSE vs rollout step (error accumulation), and returns the per-step RMSE plus
    the full predicted / ground-truth fields (for the GIF).

    Returns
    -------
    rmse_list : list[float]                       velocity RMSE per step
    preds     : list[np.ndarray[N,3]]  (u, v, p)  prediction in PHYSICAL units
    exacts    : list[np.ndarray[N,3]]  (u, v, p)  ground-truth in PHYSICAL units
    """
    ns = {k: torch.as_tensor(v).to(device) for k, v in ds.node_stats.items()}
    mask = ds.rollout_mask[case_idx].to(device).bool().view(-1, 1)  # [N,1], True=update
    mask2 = mask.repeat(1, 2)
    spc = ds.num_steps - 1
    rmse_list, preds, exacts = [], [], []
    cur_vel_norm = None
    for t in range(steps):
        g = ds[case_idx * spc + t].to(device)
        gt_vel_t = g.x[:, 0:2] * ns["velocity_std"] + ns["velocity_mean"]
        exact_next = gt_vel_t + (g.y[:, 0:2] * ns["velocity_diff_std"] + ns["velocity_diff_mean"])
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
        m = mask.view(-1)
        rmse = torch.sqrt(((new_vel[m] - exact_next[m]) ** 2).mean()).item()
        rmse_list.append(rmse)
        preds.append(torch.cat([new_vel, pred_p.unsqueeze(1)], dim=1).cpu().numpy())
        exacts.append(torch.cat([exact_next, exact_p.unsqueeze(1)], dim=1).cpu().numpy())

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
    return rmse_list, preds, exacts


def make_gif(pos, cells, preds, exacts, var_idx, var_name, save_path, frame_skip):
    """Animated field GIF, 2 rows (top: prediction, bottom: ground-truth), in the
    style of the stock inference.py. Color scale is fixed to the GT range across
    all frames so the animation is comparable frame-to-frame."""
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], cells)
    pred_i = [p[:, var_idx] for p in preds]
    exact_i = [e[:, var_idx] for e in exacts]
    vmin = float(np.min([e.min() for e in exact_i]))
    vmax = float(np.max([e.max() for e in exact_i]))

    plt.rcParams["image.cmap"] = "inferno"
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    fig.set_facecolor("black")
    for a in ax:
        a.set_facecolor("black")

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
    print(f"  Saved: {save_path}")


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
    p.add_argument("--gif", action="store_true", default=False,
                   help="save animated field GIF(s) from the rollout (implies "
                        "rollout; needs --split test). One GIF per --gif_fields.")
    p.add_argument("--gif_fields", type=str, nargs="+", default=["u", "v", "p"],
                   choices=list(VAR_ID.keys()),
                   help="which field(s) to animate: u, v, p (default all three)")
    p.add_argument("--frame_skip", type=int, default=1,
                   help="use every N-th rollout frame in the GIF (default 1)")
    p.add_argument("--gif_dir", type=str, default="./animations",
                   help="output dir for the GIFs (default ./animations)")
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

    if args.rollout > 0 or args.gif:
        if not ds.rollout_mask:
            raise ValueError("rollout/gif needs rollout_mask; use --split test")
        steps = args.rollout if args.rollout > 0 else args.num_steps - 1
        _, preds, exacts = run_rollout(model, ds, cache, args.case_idx, steps,
                                       thr, device, args.out_dir)
        if args.gif:
            os.makedirs(args.gif_dir, exist_ok=True)
            pos = cache["pos"].cpu().numpy()
            cells = np.asarray(ds.cells[args.case_idx])
            for f in args.gif_fields:
                save = os.path.join(args.gif_dir,
                                    f"amr_m4gn_case{args.case_idx}_{f}.gif")
                make_gif(pos, cells, preds, exacts, VAR_ID[f], f, save,
                         args.frame_skip)
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
