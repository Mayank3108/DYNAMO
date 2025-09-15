# train/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config as cfg
from dataset.assemblyDataset import GearAssemblyDataset
from models.dynamo import Dynamo
from utils.lossFunctions import compute_losses

# ----------- Setup -----------
def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = GearAssemblyDataset(cfg.DATA_BASE_PATH, cfg.TRAIN_LIST_PATH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    model = Dynamo().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    ckpt_path = getattr(cfg, "CHECKPOINT_PATH", "./checkpoints/DYNAMO_checkpoint.pt")

    return model, train_loader, optimizer, scheduler, device, ckpt_path

# ----------- Training Loop -----------
def train():
    model, train_loader, optimizer, scheduler, device, ckpt_path = setup()
    print(f"[ckpt] saving best to: {ckpt_path}")
    torch.autograd.set_detect_anomaly(True)

    best_loss = float("inf")

    loss_config = {
        'use_l2': True,
        'use_lav_const': True,
        'use_geodesic_rot': True,
        'lambda_trans': cfg.LAMBDA_TRANS,
        'lambda_rot': cfg.LAMBDA_ROT,
        'lambda_lav_const': cfg.LAMBDA_LAV_CONST,
    }

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        logs = {}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            pc = batch["point_cloud"].permute(0, 1, 3, 2)   # (B, P, 3, N)
            lav_gt = batch["lav"].permute(0, 2, 1, 3)       # (B, P, T, 6)
            mask = batch["mask"]
            coupling_matrix = batch["coupling_matrix"]

            # Forward
            lav_pred = model(pc, coupling_matrix)            # (B, P, T, 6)

            # Loss
            total_loss, loss_dict = compute_losses(
                lav_pred=lav_pred,
                lav_gt=lav_gt,
                pc=pc,               
                mask=mask,
                loss_config=loss_config
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.CLIP_GRAD_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
            optimizer.step()

            epoch_loss += total_loss.item()
            for key, val in loss_dict.items():
                logs[key] = logs.get(key, 0.0) + float(val.item())

        # Averages
        avg_loss = epoch_loss / max(1, len(train_loader))
        for k in logs:
            logs[k] /= max(1, len(train_loader))

        if epoch % 10 == 0 or epoch == cfg.NUM_EPOCHS - 1:
            line = f"Epoch {epoch:03d} | LR: {optimizer.param_groups[0]['lr']:.2e} | Total: {avg_loss:.6f}"
            for k, v in logs.items():
                line += f" | {k}: {v:.6f}"
            print(line)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step(avg_loss)

# ----------- Run -----------
if __name__ == "__main__":
    train()
