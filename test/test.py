# File: test/test.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import config as cfg
from dataset.assemblyDataset import GearAssemblyDataset
from models.dynamo import Dynamo
from utils.metrics import lav_to_se3_sequence, kabsch_rotation_angle_deg


def print_stats(label, values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        print(f"{label}: No data")
    else:
        print(f"{label}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}, N = {len(values)}")


@torch.no_grad()
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GearAssemblyDataset(cfg.DATA_BASE_PATH, cfg.TEST_LIST_PATH)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Checkpoint
    ckpt_path = getattr(cfg, "CHECKPOINT_PATH_V4", getattr(cfg, "CHECKPOINT_PATH", None))
    if ckpt_path is None:
        raise ValueError("No checkpoint path defined. Add CHECKPOINT_PATH_V4 or CHECKPOINT_PATH to train_config.")

    model = Dynamo().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    NUM_FRAMES = cfg.NUM_FRAMES

    assemblies = []

    range_buckets = [(0, 5, "1-5"), (0, 15, "1-15"),
                     (0, 25, "1-25"), (0, 35, "1-35")]
    window_buckets = [(0, 5, "1-5"), (5, 15, "6-15"),
                      (15, 25, "16-25"), (25, 35, "26-35")]

    range_rot_errors = {tag: [] for _, _, tag in range_buckets}
    range_trans_errors = {tag: [] for _, _, tag in range_buckets}
    moving_window_rot_errors = {tag: [] for _, _, tag in window_buckets}
    moving_window_trans_errors = {tag: [] for _, _, tag in window_buckets}

    part_count_rot_errors = {p: [] for p in range(2, 8)}
    part_count_trans_errors = {p: [] for p in range(2, 8)}

    temporal_rot_sum = {p: np.zeros(NUM_FRAMES, dtype=float) for p in range(2, 8)}
    temporal_trans_sum = {p: np.zeros(NUM_FRAMES, dtype=float) for p in range(2, 8)}
    temporal_counts = {p: np.zeros(NUM_FRAMES, dtype=float) for p in range(2, 8)}

    overall_rot_errors = []
    overall_trans_errors = []

    for idx, batch in enumerate(loader):
        filename = dataset.file_names[idx]
        mask = batch["mask"][0]
        num_parts = int(torch.sum(mask).item())

        # Inputs
        pc_parts = batch["point_cloud"][0][mask].cpu().numpy()             # (P_real, N, 3)
        lav_gt   = batch["lav"][0].permute(1, 0, 2)[mask].to(device)       # (P_real, T, 6)
        coupling = batch["coupling_matrix"].to(device)

        # Forward
        pc_input = batch["point_cloud"].to(device).permute(0, 1, 3, 2)     # (1, P, 3, N)
        lav_pred = model(pc_input, coupling)[0][mask]                      # (P_real, T, 6)

        # To SE(3) sequences
        se3_gt   = lav_to_se3_sequence(lav_gt)    # (P_real, T+1, 4, 4)
        se3_pred = lav_to_se3_sequence(lav_pred)  # (P_real, T+1, 4, 4)

        frame_rot_sum = np.zeros(NUM_FRAMES, dtype=float)
        frame_trans_sum = np.zeros(NUM_FRAMES, dtype=float)
        frame_counts = np.zeros(NUM_FRAMES, dtype=float)

        for pid in range(pc_parts.shape[0]):
            pc0 = pc_parts[pid]                                  # (N,3)
            N = pc0.shape[0]
            if N < 3:
                continue

            homo = np.hstack([pc0, np.ones((N, 1), dtype=np.float64)])     # (N,4)

            for t in range(1, NUM_FRAMES + 1):
                T_gt_prev = se3_gt[pid, t - 1].cpu().numpy()
                T_gt_curr = se3_gt[pid, t].cpu().numpy()
                gt_prev = (T_gt_prev @ homo.T).T[:, :3]                     # (N,3)
                gt_curr = (T_gt_curr @ homo.T).T[:, :3]                     # (N,3)

                T_pr_prev = se3_pred[pid, t - 1].cpu().numpy()
                T_pr_curr = se3_pred[pid, t].cpu().numpy()
                pr_prev = (T_pr_prev @ homo.T).T[:, :3]
                pr_curr = (T_pr_curr @ homo.T).T[:, :3]

                # ---- Rotation (Kabsch angle on centered clouds) ----
                # Centering removes translation; angle is rigid rotation magnitude
                gt_prev_c = gt_prev - gt_prev.mean(axis=0, keepdims=True)
                gt_curr_c = gt_curr - gt_curr.mean(axis=0, keepdims=True)
                pr_prev_c = pr_prev - pr_prev.mean(axis=0, keepdims=True)
                pr_curr_c = pr_curr - pr_curr.mean(axis=0, keepdims=True)

                ang_gt_deg, _ = kabsch_rotation_angle_deg(gt_prev_c, gt_curr_c)
                ang_pr_deg, _ = kabsch_rotation_angle_deg(pr_prev_c, pr_curr_c)

                rot_error_deg = abs(ang_pr_deg - ang_gt_deg)

                # ---- Translation (unchanged): mean point distance between GT and Pred at t ----
                trans_error = np.mean(np.linalg.norm(gt_curr - pr_curr, axis=1))

                overall_rot_errors.append(rot_error_deg)
                overall_trans_errors.append(trans_error)

                frame_rot_sum[t - 1]  += rot_error_deg
                frame_trans_sum[t - 1] += trans_error
                frame_counts[t - 1]    += 1

        with np.errstate(invalid='ignore', divide='ignore'):
            frame_rot_avg = np.divide(frame_rot_sum, frame_counts,
                                      out=np.full_like(frame_rot_sum, np.nan),
                                      where=frame_counts > 0)
            frame_trans_avg = np.divide(frame_trans_sum, frame_counts,
                                        out=np.full_like(frame_trans_sum, np.nan),
                                        where=frame_counts > 0)

        asm_mean_rot = np.nanmean(frame_rot_avg)
        asm_mean_trans = np.nanmean(frame_trans_avg)

        assemblies.append({
            "name": filename,
            "part_count": num_parts,
            "mean_rot": float(asm_mean_rot) if asm_mean_rot == asm_mean_rot else np.nan,
            "mean_trans": float(asm_mean_trans) if asm_mean_trans == asm_mean_trans else np.nan,
            "frame_rot": frame_rot_avg,
            "frame_trans": frame_trans_avg
        })

        # Range buckets
        for start, end, tag in range_buckets:
            r = np.nanmean(frame_rot_avg[start:end])
            t = np.nanmean(frame_trans_avg[start:end])
            range_rot_errors[tag].append(r)
            range_trans_errors[tag].append(t)

        # Moving windows
        for start, end, tag in window_buckets:
            r = np.nanmean(frame_rot_avg[start:end])
            t = np.nanmean(frame_trans_avg[start:end])
            moving_window_rot_errors[tag].append(r)
            moving_window_trans_errors[tag].append(t)

        # Part-count groupings + temporal curves
        if 2 <= num_parts <= 7:
            if asm_mean_rot == asm_mean_rot:
                part_count_rot_errors[num_parts].append(asm_mean_rot)
            if asm_mean_trans == asm_mean_trans:
                part_count_trans_errors[num_parts].append(asm_mean_trans)

            valid_mask = ~np.isnan(frame_rot_avg)
            temporal_rot_sum[num_parts][valid_mask]   += frame_rot_avg[valid_mask]
            temporal_trans_sum[num_parts][valid_mask] += frame_trans_avg[valid_mask]
            temporal_counts[num_parts][valid_mask]    += 1


    print("\n=== PER-ASSEMBLY MEAN ERRORS (Rotation°, Translation units) ===")
    if len(assemblies) == 0:
        print("No assemblies.")
    else:
        for i, rec in enumerate(assemblies, 1):
            name = os.path.basename(str(rec["name"]))
            pc = rec["part_count"]
            r, t = rec["mean_rot"], rec["mean_trans"]
            r_str = f"{r:.4f}" if r == r else "nan"
            t_str = f"{t:.4f}" if t == t else "nan"
            print(f"{i:02d}. {name} (P={pc}): rot={r_str}°, trans={t_str}")

    mean_of_means_rot = np.nanmean([rec["mean_rot"] for rec in assemblies]) if assemblies else np.nan
    mean_of_means_trans = np.nanmean([rec["mean_trans"] for rec in assemblies]) if assemblies else np.nan
    print("\n=== OVERALL MEAN OF ASSEMBLY MEANS ===")
    print(f"Rotation mean-of-means: {mean_of_means_rot:.4f}°" if mean_of_means_rot == mean_of_means_rot else "Rotation mean-of-means: No data")
    print(f"Translation mean-of-means: {mean_of_means_trans:.4f}" if mean_of_means_trans == mean_of_means_trans else "Translation mean-of-means: No data")

    print("\n=== FRAME RANGE ERRORS (mean over assemblies' frame-range means) ===")
    for _, _, tag in range_buckets:
        r_mean = np.nanmean(range_rot_errors[tag]) if len(range_rot_errors[tag]) else np.nan
        t_mean = np.nanmean(range_trans_errors[tag]) if len(range_trans_errors[tag]) else np.nan
        print(f"{tag} - Rotation: {'nan' if r_mean != r_mean else f'{r_mean:.4f}°'}")
        print(f"{tag} - Translation: {'nan' if t_mean != t_mean else f'{t_mean:.4f}'}")

    print("\n=== FRAME WINDOW ERRORS (mean over assemblies' window means) ===")
    for _, _, tag in window_buckets:
        r_mean = np.nanmean(moving_window_rot_errors[tag]) if len(moving_window_rot_errors[tag]) else np.nan
        t_mean = np.nanmean(moving_window_trans_errors[tag]) if len(moving_window_trans_errors[tag]) else np.nan
        print(f"{tag} - Rotation: {'nan' if r_mean != r_mean else f'{r_mean:.4f}°'}")
        print(f"{tag} - Translation: {'nan' if t_mean != t_mean else f'{t_mean:.4f}'}")

    print("\n=== PART COUNT ERRORS (mean over assembly means in each group) ===")
    for p in range(2, 8):
        r_mean = np.nanmean(part_count_rot_errors[p]) if part_count_rot_errors[p] else np.nan
        t_mean = np.nanmean(part_count_trans_errors[p]) if part_count_trans_errors[p] else np.nan
        print(f"{p} Parts - Rotation: {'nan' if r_mean != r_mean else f'{r_mean:.4f}°'}")
        print(f"{p} Parts - Translation: {'nan' if t_mean != t_mean else f'{t_mean:.4f}'}")

    print("\n=== TEMPORAL ERROR CURVES (per part-count; frame-wise means) ===")
    for p in range(2, 8):
        if np.any(temporal_counts[p] > 0):
            rot_curve = np.divide(
                temporal_rot_sum[p], temporal_counts[p],
                out=np.full_like(temporal_rot_sum[p], np.nan),
                where=temporal_counts[p] > 0
            )
            trans_curve = np.divide(
                temporal_trans_sum[p], temporal_counts[p],
                out=np.full_like(temporal_trans_sum[p], np.nan),
                where=temporal_counts[p] > 0
            )
            print(f"{p} Parts - Rotation (framewise): {np.round(rot_curve, 4).tolist()}")
            print(f"{p} Parts - Translation (framewise): {np.round(trans_curve, 4).tolist()}")
        else:
            print(f"{p} Parts - Rotation (framewise): []")
            print(f"{p} Parts - Translation (framewise): []")


if __name__ == "__main__":
    evaluate()
