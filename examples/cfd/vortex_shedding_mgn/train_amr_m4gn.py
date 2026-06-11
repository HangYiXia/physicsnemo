"""
train_amr_m4gn.py — minimal single-case overfit trainer for AMRM4GN (M4)
========================================================================
Goal of M4 (Design Doc 8 / 7.7): overfit a SINGLE case so the loss drops
monotonically toward ~0 and the prediction matches the ground truth. This is a
slim argparse script (no DDP/AMP/wandb) focused on that integration check; the
full hydra/DDP trainer is M5/M7.

Pipeline per step:
    graph(x[N,6], edge_attr, edge_index, pos, y[N,3]) + per-case geometry cache
    -> AMRM4GN.forward -> pred[N,3] -> per-channel NMSE vs graph.y

Usage:
    cd examples/cfd/vortex_shedding_mgn
    python train_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --split test --case_idx 0 --num_steps 50 --epochs 300 --omega_thresh 30
"""

from __future__ import annotations

import argparse

import torch

from data_amr import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from preprocess_partitions import build_cache


def per_channel_nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Mean over channels of MSE_c / mean(target_c^2) (Design Doc 4.9)."""
    num = ((pred - target) ** 2).mean(dim=0)
    den = (target ** 2).mean(dim=0) + eps
    return (num / den).mean()


def move_cache(cache: dict, device):
    out = {}
    for k, v in cache.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv)
                      for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = [vv.to(device) if torch.is_tensor(vv) else vv for vv in v]
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train",
                   help="Use 'train' (default): it self-computes edge/node "
                        "stats (no baseline run needed). Non-train splits "
                        "require edge_stats.json/node_stats.json in cwd.")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="Training-noise std; 0 for a clean overfit (default 0).")
    p.add_argument("--case_idx", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--omega_thresh", type=float, default=30.0,
                   help="fixed vorticity threshold for routing during overfit")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="dir with partition_cache_{split}_{idx}.pt; if unset, "
                        "the cache is built on the fly.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--K0", type=int, default=64)
    p.add_argument("--K1", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device)

    ds = VortexSheddingDatasetAMR(
        name="amr_overfit", data_dir=args.data_dir, split=args.split,
        num_samples=args.case_idx + 1, num_steps=args.num_steps,
        noise_std=args.noise_std, cache_dir=args.cache_dir,
    )

    if args.cache_dir is not None:
        cache = ds.get_cache(args.case_idx)
    else:
        print("[cache] building partition cache on the fly ...")
        cache = build_cache(args.data_dir, args.split, args.case_idx,
                            args.K0, args.K1, num_modes=6, tau=1.0)
    cache = move_cache(cache, device)

    vel_mean = ds.node_stats["velocity_mean"]
    vel_std = ds.node_stats["velocity_std"]

    model = AMRM4GN(
        in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
        processor_size=args.processor_size,
        vel_mean=vel_mean, vel_std=vel_std,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    thr = {"G": float("inf"), "omega": args.omega_thresh,
           "M": float("inf"), "S": float("inf")}

    # indices belonging to the chosen case (gidx == case_idx)
    steps_per_case = args.num_steps - 1
    idxs = list(range(args.case_idx * steps_per_case,
                      (args.case_idx + 1) * steps_per_case))

    model.train()
    print(f"[overfit] case {args.case_idx}, {len(idxs)} steps, "
          f"{args.epochs} epochs, device={device}")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for idx in idxs:
            graph = ds[idx].to(device)
            opt.zero_grad()
            pred = model(graph, cache, thresholds=thr)
            loss = per_channel_nmse(pred, graph.y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(idxs)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  NMSE {epoch_loss:.4e}")
    print("done.")


if __name__ == "__main__":
    main()
