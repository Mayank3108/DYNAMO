# test/visualize_prediction.py
import os
import torch
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader, Subset
from liegroups.torch import SE3

from config import config as cfg
from dataset.assemblyDataset import GearAssemblyDataset
from models.dynamo import Dynamo

# === SETTINGS ===
target_name = "Assembly4076.npz"
PLOT_GT = True  # Set to False if you want to plot only predicted trajectory, TRUE to plot both GT and Predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_frames = cfg.NUM_FRAMES
base_path = cfg.DATA_BASE_PATH
list_file = cfg.TEST_LIST_PATH

# === LOAD DATA ===
dataset = GearAssemblyDataset(base_path, list_file)
assert target_name in dataset.file_names, f"{target_name} not found in {list_file}"
target_idx = dataset.file_names.index(target_name)
loader = DataLoader(Subset(dataset, [target_idx]), batch_size=1, shuffle=False)
batch = next(iter(loader))

# === LOAD MODEL ===
model = Dynamo().to(device)
model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location=device))
model.eval()

# === EXTRACT INPUTS ===
pc = batch["point_cloud"].to(device).permute(0, 1, 3, 2)        # (1, P, 3, N)
lav_gt = batch["lav"].to(device).permute(0, 2, 1, 3)[0]         # (P, T, 6)
mask = batch["mask"][0]                                         # (P,)
coupling = batch["coupling_matrix"].to(device)

# === PREDICT ===
with torch.no_grad():
    lav_pred = model(pc, coupling)[0]                           # (P, T, 6)

# === CONVERT LAVs to SE3 sequences ===
def lav_to_se3_sequence(lav):
    P, T, _ = lav.shape
    traj = torch.zeros(P, T + 1, 4, 4, device=lav.device)
    traj[:, 0] = torch.eye(4, device=lav.device)
    for t in range(T):
        dT = SE3.exp(lav[:, t])
        traj[:, t + 1] = torch.bmm(traj[:, t], dT.as_matrix())
    return traj  # (P, T+1, 4, 4)



lav_gt = lav_gt[mask]
lav_pred = lav_pred[mask]
se3_gt = lav_to_se3_sequence(lav_gt)
se3_pred = lav_to_se3_sequence(lav_pred)

# === TRANSFORM POINT CLOUD ===
pc_parts = batch["point_cloud"][0][mask].cpu().numpy()  # (P_real, N, 3)
points_gt_all = {}
points_pred_all = {}

for pid in range(pc_parts.shape[0]):
    pc = pc_parts[pid]
    homo = np.hstack([pc, np.ones((pc.shape[0], 1))])
    points_gt_all[pid] = [(se3_gt[pid, f].cpu().numpy() @ homo.T).T[:, :3] for f in range(num_frames + 1)]
    points_pred_all[pid] = [(se3_pred[pid, f].cpu().numpy() @ homo.T).T[:, :3] for f in range(num_frames + 1)]

# === SCENE BOUNDS ===
all_pts = []
for pid in range(pc_parts.shape[0]):
    all_pts.append(pc_parts[pid])
    all_pts += points_pred_all[pid]
    if PLOT_GT:
        all_pts += points_gt_all[pid]

combined = np.vstack(all_pts)
x_mid, y_mid, z_mid = np.mean(combined, axis=0)
max_range = np.max(np.ptp(combined, axis=0)) * 1.2

# === ANIMATION FRAMES ===
frames = []
slider_steps = []

for t in range(num_frames + 1):
    frame_data = []

    if PLOT_GT:
        gt_data = [go.Scatter3d(
            x=points_gt_all[pid][t][:, 0],
            y=points_gt_all[pid][t][:, 1],
            z=points_gt_all[pid][t][:, 2],
            mode="markers",
            marker=dict(size=2, color="green", opacity=0.6),
            showlegend=False
        ) for pid in points_gt_all]
        frames.append(go.Frame(name=f"GT_Frame_{t}", data=gt_data))
        slider_steps.append(dict(
            method="animate",
            args=[[f"GT_Frame_{t}"], {"mode": "immediate", "frame": {"duration": 150, "redraw": True}}],
            label=f"GT {t}"
        ))

        # === Pred Frame ===
    pred_data = [go.Scatter3d(
        x=points_pred_all[pid][t][:, 0],
        y=points_pred_all[pid][t][:, 1],
        z=points_pred_all[pid][t][:, 2],
        mode="markers",
        marker=dict(size=2, color="red", opacity=0.6),
        showlegend=False
    ) for pid in points_pred_all]
    frames.append(go.Frame(name=f"Pred_Frame_{t}", data=pred_data))
    slider_steps.append(dict(
        method="animate",
        args=[[f"Pred_Frame_{t}"], {"mode": "immediate", "frame": {"duration": 150, "redraw": True}}],
        label=f"Pred {t}"
    ))
# === INITIAL FRAME ===
initial_data = [go.Scatter3d(x=pc_parts[pid][:, 0], y=pc_parts[pid][:, 1], z=pc_parts[pid][:, 2],
                             mode="markers", marker=dict(size=2, opacity=0.6), showlegend=False)
                for pid in range(pc_parts.shape[0])]

# === PLOT ===
fig = go.Figure(
    data=initial_data,
    frames=frames,
    layout=go.Layout(
        title=f"{target_name}: {'GT (green) vs Pred (red)' if PLOT_GT else 'Prediction Only'}",
        scene=dict(
            xaxis=dict(range=[x_mid - max_range / 2, x_mid + max_range / 2], showticklabels=False),
            yaxis=dict(range=[y_mid - max_range / 2, y_mid + max_range / 2], showticklabels=False),
            zaxis=dict(range=[z_mid - max_range / 2, z_mid + max_range / 2], showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[
            dict(type="buttons", showactive=False,
                 buttons=[
                     dict(label="Play", method="animate",
                          args=[None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause", method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                 ])
        ],
        sliders=[dict(steps=slider_steps, active=0)]
    )
)

fig.show()
