"""
run_ablation.py — M6 module-ablation study for AMR-M4GN (Design Doc 6.5)
=======================================================================
Trains AMR-M4GN under a fixed budget for several module-ablation configs and
evaluates each on the SAME test-case rollout, producing one comparison table
(printed + CSV) and a bar chart (13_ablation.png). All configs share the data,
budget and eval, so the per-row delta is the net contribution of one module.

Configs (Design Doc 6.5 table) — all runnable:
    full            : complete AMR-M4GN (δ=0, no virtual step — the M5 baseline)
    w/o AMR         : use_amr=False  -> every L1 segment stays fine (fixed K=K1)
    w/o Transformer : use_transformer=False -> decode from the GNN only (~MGN)
    w/o Modal       : geometry-only partition cache (preprocess --no_modal)
    w/o RWSE        : use_rwse=False -> zero the segment positional encoding
    proc7           : processor_size=7 (vs 15) -> GNN depth收益 vs 过平滑
    w/ overlap      : use_overlap=True  -> δ=1 1-ring halo in pool/dispatch
    w/ virtual      : use_virtual_step=True -> route on the virtual velocity field

Note on overlap/virtual rows: the §6.5 table frames them as "w/o δ overlap" /
"w/o virtual step". Since the M5 baseline `full` is ALREADY δ=0 and without the
virtual step, these two rows measure the *marginal effect of ADDING* the module
(lower RMSE than `full` => the module helps and should be kept).

Prerequisite caches (run preprocess first):
    python preprocess_partitions.py --split train --num_cases N            # modal
    python preprocess_partitions.py --split test  --num_cases 1            # modal
    python preprocess_partitions.py --split train --num_cases N --no_modal # for w/o Modal
    python preprocess_partitions.py --split test  --num_cases 1 --no_modal # for w/o Modal

Usage:
    python run_ablation.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow \
        --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --batch_size 2 \
        --epochs 200 --noise_std 0.02 --omega_thresh 8.9 \
        --test_case 0 --rollout 80
"""

from __future__ import annotations

import os
import csv
import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.vortex import VortexSheddingDatasetAMR
from amr_m4gn.model import AMRM4GN
from scripts.train_amr_m4gn import per_channel_nmse, move_cache
from scripts.compare_baselines import rollout_eval


# config name -> (model kwargs override, processor_size, cache suffix, runnable, note)
CONFIGS = [
    ("full",            dict(),                          15, "",        True,  "完整模型 (δ=0, 无虚拟步)"),
    ("w/o AMR",         dict(use_amr=False),             15, "",        True,  "固定 K=K1，不折叠"),
    ("w/o Transformer", dict(use_transformer=False),     15, "",        True,  "仅 GNN，无全局注意力"),
    ("w/o Modal",       dict(),                          15, "_nomodal",True,  "几何-only 分区缓存"),
    ("w/o RWSE",        dict(use_rwse=False),            15, "",        True,  "去段级位置编码"),
    ("proc7",           dict(),                           7, "",        True,  "GNN 7 步 vs 15 步"),
    ("w/ overlap",      dict(use_overlap=True),          15, "",        True,  "δ=1 一圈邻居重叠 (相对 full 的增益)"),
    ("w/ virtual",      dict(use_virtual_step=True),     15, "",        True,  "虚拟步路由 (相对 full 的增益)"),
]


def load_caches(cache_dir, split, n, suffix, device):
    out = {}
    for g in range(n):
        path = os.path.join(cache_dir, f"partition_cache_{split}_{g}{suffix}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing cache {path}\nRun preprocess_partitions.py "
                f"{'--no_modal ' if suffix else ''}for split={split}.")
        out[g] = move_cache(torch.load(path, weights_only=False), device)
    return out


def train_one(model, loader, caches, epochs, lr, lr_decay, thr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    model.train()
    last = float("nan")
    for epoch in range(epochs):
        ep, nb = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            cs = [caches[int(g)] for g in batch.gidx]
            opt.zero_grad()
            pred = model(batch, cs, thresholds=thr)
            loss = per_channel_nmse(pred, batch.y)
            loss.backward()
            opt.step()
            ep += loss.item(); nb += 1
        sched.step()
        last = ep / max(nb, 1)
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--num_cases", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.9999991)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--omega_thresh", type=float, default=8.9)
    p.add_argument("--test_split", type=str, default="test")
    p.add_argument("--test_case", type=int, default=0)
    p.add_argument("--test_num_steps", type=int, default=90)
    p.add_argument("--rollout", type=int, default=80)
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="run only these config names (default: all runnable)")
    p.add_argument("--out_dir", type=str, default="./inference_vis")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    thr = {"G": float("inf"), "omega": args.omega_thresh,
           "M": float("inf"), "S": float("inf")}

    # train / test datasets (modal). node_stats/cells/rollout_mask come from here.
    ds_tr = VortexSheddingDatasetAMR(
        name="abl_train", data_dir=args.data_dir, split="train",
        num_samples=args.num_cases, num_steps=args.num_steps,
        noise_std=args.noise_std, cache_dir=args.cache_dir)
    ds_te = VortexSheddingDatasetAMR(
        name="abl_test", data_dir=args.data_dir, split=args.test_split,
        num_samples=args.test_case + 1, num_steps=args.test_num_steps,
        noise_std=0.0, cache_dir=args.cache_dir)
    loader = PyGDataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    vel_mean, vel_std = ds_tr.node_stats["velocity_mean"], ds_tr.node_stats["velocity_std"]
    ns = {k: torch.as_tensor(v).to(device) for k, v in ds_te.node_stats.items()}
    mask = ds_te.rollout_mask[args.test_case]

    # cache cache-sets so we don't reload per config
    cache_sets = {}  # suffix -> (train_caches, test_cache)

    rows = []
    todo = [c for c in CONFIGS if (args.only is None and c[4]) or
            (args.only is not None and c[0] in args.only)]
    for name, kw, proc, suffix, runnable, note in todo:
        if not runnable:
            print(f"[ablation] SKIP {name}: {note}")
            rows.append((name, None, None, None, note))
            continue
        if suffix not in cache_sets:
            cache_sets[suffix] = (
                load_caches(args.cache_dir, "train", args.num_cases, suffix, device),
                load_caches(args.cache_dir, args.test_split, args.test_case + 1, suffix, device)[args.test_case],
            )
        train_caches, test_cache = cache_sets[suffix]

        print(f"[ablation] train {name} (proc={proc}, suffix='{suffix or '-'}', {note}) ...")
        torch.manual_seed(0)
        model = AMRM4GN(in_nodes=6, in_edges=3, out_dim=3, hidden=args.hidden,
                        processor_size=proc, vel_mean=vel_mean, vel_std=vel_std,
                        **kw).to(device)
        n_par = sum(q.numel() for q in model.parameters()) / 1e6
        train_nmse = train_one(model, loader, train_caches, args.epochs,
                               args.lr, args.lr_decay, thr, device)
        model.eval()
        rmse, _ = rollout_eval(lambda g: model(g, test_cache, thresholds=thr),
                               ds_te, args.test_case, args.rollout, ns, mask, device)
        mean_rmse = float(np.mean(rmse))
        print(f"  -> params {n_par:.2f}M | train NMSE {train_nmse:.4e} | "
              f"rollout mean RMSE {mean_rmse:.4e}")
        rows.append((name, n_par, train_nmse, mean_rmse, note))

    # ---- table (print + CSV) ----
    print("\n==== M6 ablation (test case {}, rollout {} steps) ====".format(
        args.test_case, args.rollout))
    print(f"{'config':16s} {'params(M)':>10s} {'train NMSE':>12s} {'mean RMSE':>12s}  note")
    for name, par, tn, mr, note in rows:
        ps = f"{par:.2f}" if par is not None else "-"
        ts = f"{tn:.3e}" if tn is not None else "-"
        ms = f"{mr:.3e}" if mr is not None else "-"
        print(f"{name:16s} {ps:>10s} {ts:>12s} {ms:>12s}  {note}")
    csv_path = os.path.join(args.out_dir, "13_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "params_M", "train_NMSE", "rollout_mean_RMSE", "note"])
        for r in rows:
            w.writerow(r)
    print(f"Saved: {csv_path}")

    # ---- bar chart of mean rollout RMSE (runnable configs only) ----
    done = [(n, mr) for n, _, _, mr, _ in rows if mr is not None]
    if done:
        names = [d[0] for d in done]
        vals = [d[1] for d in done]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, vals, color="steelblue")
        if "full" in names:
            bars[names.index("full")].set_color("crimson")
        plt.ylabel("rollout mean velocity RMSE (test)")
        plt.title(f"M6 ablation: net contribution per module (case {args.test_case})")
        plt.xticks(rotation=20, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        save = os.path.join(args.out_dir, "13_ablation.png")
        plt.savefig(save, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save}")


if __name__ == "__main__":
    main()
