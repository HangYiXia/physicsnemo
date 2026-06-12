"""
train_amr_m4gn_full.py — multi-case batched trainer for AMRM4GN (M5 step 3)
===========================================================================
Extends the M4 overfit script (train_amr_m4gn.py, kept as-is) into a proper
training entry: multiple cases, PyG batched DataLoader (batch_size>1),
per-channel NMSE, exponential LR decay, and checkpointing. Single-machine /
single-GPU (no DDP/apex/wandb — those are added when real multi-GPU runs are
needed; the model/data pieces here are DDP-ready).

Key batching detail: the DataLoader collates B graphs into one PyG Batch
(carrying `ptr` and per-graph `gidx`); we fetch the matching list of per-case
caches via `gidx` and pass it to `AMRM4GN.forward(batch, caches)` (M5 step 2).

Prerequisite: per-case partition caches must exist (run preprocess first):
    python preprocess_partitions.py --data_dir <...> --split train \
        --num_cases <N> --out_dir ./amr_cache

Usage (small smoke run):
    python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --batch_size 2 \
        --epochs 50 --omega_thresh 30
"""

from __future__ import annotations

import os
import argparse

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.vortex import VortexSheddingDatasetAMR, make_amr_dataset
from amr_m4gn.model import AMRM4GN
from scripts.train_amr_m4gn import per_channel_nmse, move_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="vortex",
                   choices=["vortex", "eagle"],
                   help="data source: vortex (cylinder-flow) or eagle (M7)")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="dir with partition_cache_train_{gidx}.pt (preprocess first)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_cases", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.9999991)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--omega_thresh", type=float, default=30.0,
                   help="fixed vorticity threshold for routing (None ranges -> "
                        "use --sample_thresh for train-time sampling)")
    p.add_argument("--sample_thresh", action="store_true", default=False,
                   help="sample thresholds per graph (Design Doc 4.7) instead of "
                        "the fixed omega threshold")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints_amr")
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--tag", type=str, default="amr_m4gn",
                   help="checkpoint filename prefix (use to separate ablations)")
    p.add_argument("--no_amr", action="store_true", default=False,
                   help="M6 ablation: disable AMR routing (fixed K=K1)")
    p.add_argument("--no_transformer", action="store_true", default=False,
                   help="M6 ablation: disable macro Transformer (GNN only)")
    p.add_argument("--no_rwse", action="store_true", default=False,
                   help="M6 ablation: zero the segment-level RWSE PE")
    p.add_argument("--use_overlap", action="store_true", default=False,
                   help="M6 ablation: enable δ=1 (1-ring halo) segment overlap")
    p.add_argument("--use_virtual", action="store_true", default=False,
                   help="M6 ablation: route on the forward-Euler virtual field")
    p.add_argument("--clip", type=float, default=1.0,
                   help="grad-norm clip (0 to disable). Default 1.0 — required, "
                        "without it batched AMR-M4GN training can diverge after a "
                        "few epochs (NMSE jumps to ~1 = collapsed-to-zero output).")
    p.add_argument("--warmup", type=int, default=5,
                   help="linear-LR warmup epochs from lr/10 -> lr (helps the "
                        "early-epoch spike that triggers divergence).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    ds = make_amr_dataset(
        args.dataset, name="amr_train", data_dir=args.data_dir, split=args.split,
        num_samples=args.num_cases, num_steps=args.num_steps,
        noise_std=args.noise_std, cache_dir=args.cache_dir,
    )
    loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # preload all per-case caches onto the device
    caches = {g: move_cache(ds.get_cache(g), device) for g in range(args.num_cases)}

    vel_mean = ds.node_stats["velocity_mean"]
    vel_std = ds.node_stats["velocity_std"]
    model = AMRM4GN(
        in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
        processor_size=args.processor_size, vel_mean=vel_mean, vel_std=vel_std,
        use_amr=not args.no_amr, use_transformer=not args.no_transformer,
        use_rwse=not args.no_rwse, use_overlap=args.use_overlap,
        use_virtual_step=args.use_virtual,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)

    thr = None if args.sample_thresh else {
        "G": float("inf"), "omega": args.omega_thresh,
        "M": float("inf"), "S": float("inf")}

    model.train()
    print(f"[train] {args.num_cases} cases x {args.num_steps-1} steps, "
          f"batch_size={args.batch_size}, {args.epochs} epochs, device={device}, "
          f"clip={args.clip}, warmup={args.warmup}")
    base_lr = args.lr
    nan_skipped_total = 0
    for epoch in range(args.epochs):
        # linear LR warmup for the first `warmup` epochs (lr/10 -> lr)
        if args.warmup > 0 and epoch < args.warmup:
            warm = (epoch + 1) / max(args.warmup, 1)
            wlr = base_lr * (0.1 + 0.9 * warm)
            for pg in opt.param_groups:
                pg["lr"] = wlr
        epoch_loss, nb, nan_skipped, gn_sum = 0.0, 0, 0, 0.0
        for batch in loader:
            batch = batch.to(device)
            cs = [caches[int(g)] for g in batch.gidx]
            opt.zero_grad()
            pred = model(batch, cs, thresholds=thr)
            loss = per_channel_nmse(pred, batch.y)
            # NaN/inf guard: a single bad batch can poison the weights forever
            if not torch.isfinite(loss):
                nan_skipped += 1
                continue
            loss.backward()
            if args.clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                gn_sum += float(gn)
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        # only step the exponential decay AFTER warmup ends
        if args.warmup == 0 or epoch >= args.warmup:
            sched.step()
        nan_skipped_total += nan_skipped
        epoch_loss /= max(nb, 1)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            avg_gn = gn_sum / max(nb, 1) if args.clip > 0 else float("nan")
            extra = f"  grad_norm {avg_gn:.2f}" if args.clip > 0 else ""
            extra += f"  nan_skipped {nan_skipped}" if nan_skipped else ""
            print(f"epoch {epoch:4d}  NMSE {epoch_loss:.4e}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}{extra}")
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.ckpt_dir, f"{args.tag}_epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict()}, path)
    if nan_skipped_total:
        print(f"[warn] skipped {nan_skipped_total} NaN/inf batches over training.")
    print(f"done. checkpoints in {args.ckpt_dir}")


if __name__ == "__main__":
    main()
