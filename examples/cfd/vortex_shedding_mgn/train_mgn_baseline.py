"""
train_mgn_baseline.py — MGN baseline under the SAME training budget (M5)
========================================================================
A MeshGraphNet baseline trained with EXACTLY the same conditions as
train_amr_m4gn_full.py (same dataset/cases, per-channel NMSE, noise, lr, epochs,
checkpoint format), so the comparison vs AMR-M4GN is fair. The only difference
is the model: plain MeshGraphNet (node-level prediction, no AMR routing).

Usage (small-validation budget):
    python train_mgn_baseline.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02
"""

from __future__ import annotations

import os
import argparse

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from train_amr_m4gn import per_channel_nmse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_cases", type=int, default=20)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.9999991)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--processor_size", type=int, default=15)
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints_mgn")
    p.add_argument("--ckpt_every", type=int, default=20)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    ds = VortexSheddingDataset(
        name="mgn_baseline", data_dir=args.data_dir, split=args.split,
        num_samples=args.num_cases, num_steps=args.num_steps,
        noise_std=args.noise_std,
    )
    loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = MeshGraphNet(
        input_dim_nodes=6, input_dim_edges=3, output_dim=3,
        processor_size=args.processor_size,
        hidden_dim_processor=args.hidden,
        hidden_dim_node_encoder=args.hidden,
        hidden_dim_edge_encoder=args.hidden,
        hidden_dim_node_decoder=args.hidden,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)

    nparams = sum(p.numel() for p in model.parameters())
    model.train()
    print(f"[MGN baseline] {args.num_cases} cases x {args.num_steps-1} steps, "
          f"batch={args.batch_size}, {args.epochs} epochs, params={nparams/1e6:.2f}M, "
          f"device={device}")
    for epoch in range(args.epochs):
        epoch_loss, nb = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch.x, batch.edge_attr, batch)
            loss = per_channel_nmse(pred, batch.y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        sched.step()
        epoch_loss /= max(nb, 1)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  NMSE {epoch_loss:.4e}  lr {sched.get_last_lr()[0]:.2e}")
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt": opt.state_dict()},
                       os.path.join(args.ckpt_dir, f"mgn_epoch{epoch}.pt"))
    print(f"done. checkpoints in {args.ckpt_dir}")


if __name__ == "__main__":
    main()
