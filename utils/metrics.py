# utils/lav_loss_utils.py
import torch
from liegroups.torch import SE3
import numpy as np

def l2_loss(pred, gt, mask=None):
    """
    Generic L2 loss function.
    pred, gt: (..., 3)
    mask: optional (...,) boolean or float mask
    Returns: scalar loss
    """
    loss = (pred - gt).pow(2).sum(-1)  # (...,)

    if mask is not None:
        if mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = loss.mean()

    return loss

# ===========================
# Evaluation helpers
# ===========================

EPS = 1e-9

def lav_to_se3_sequence(lav: torch.Tensor) -> torch.Tensor:
    """
    lav: (P, T, 6) -> traj: (P, T+1, 4, 4)
    Accumulates per-frame twists (se(3)) into absolute SE(3) per part.
    """
    P, T, _ = lav.shape
    traj = torch.zeros(P, T + 1, 4, 4, device=lav.device, dtype=lav.dtype)
    traj[:, 0] = torch.eye(4, device=lav.device, dtype=lav.dtype)
    for t in range(T):
        dT = SE3.exp(lav[:, t])
        traj[:, t + 1] = torch.bmm(traj[:, t], dT.as_matrix())
    return traj

def transform_points_by_se3_sequence(points_np: np.ndarray, se3_traj_pt: torch.Tensor) -> list:
    """
    points_np: (N,3) numpy
    se3_traj_pt: (T+1, 4,4) torch
    returns: list of (N,3) numpy for frames 0..T
    """
    N = points_np.shape[0]
    homo = np.hstack([points_np, np.ones((N, 1), dtype=np.float64)])  # (N,4)
    out = []
    for f in range(se3_traj_pt.shape[0]):
        T = se3_traj_pt[f].detach().cpu().numpy()
        cur = (T @ homo.T).T[:, :3]
        out.append(cur)
    return out

def kabsch_rotation_angle_deg(Pc: np.ndarray, Qc: np.ndarray):
    """
    Best-fit rotation angle (deg) and axis between centered point sets Pc->Qc.
    Pc, Qc: (N,3), mean already removed per set.
    Returns: (angle_deg: float, axis_unit: (3,) np.ndarray with NaN if angle≈0)
    """
    assert Pc.shape == Qc.shape and Pc.shape[1] == 3
    H = Pc.T @ Qc  # (3,3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1.0
        R = Vt.T @ U.T
    tr = np.trace(R)
    val = (tr - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(val)))

    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    n = np.linalg.norm(axis)
    axis_unit = axis / n if n > EPS else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return angle_deg, axis_unit

def rotation_about_given_axis_deg(Pc: np.ndarray, Qc: np.ndarray, axis: np.ndarray) -> float:
    """
    Estimate signed rotation (deg) about a GIVEN axis from centered clouds Pc->Qc.
    Steps: build ONB with 'axis', project to plane, atan2 per-point, circular mean.
    """
    a = axis.astype(np.float64)
    a = a / (np.linalg.norm(a) + EPS)

    # ONB {e1, e2, a}
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(helper, a)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(a, helper); e1 /= (np.linalg.norm(e1) + EPS)
    e2 = np.cross(a, e1);     e2 /= (np.linalg.norm(e2) + EPS)

    P2 = np.stack([Pc @ e1, Pc @ e2], axis=1)  # (N,2)
    Q2 = np.stack([Qc @ e1, Qc @ e2], axis=1)  # (N,2)

    angP = np.arctan2(P2[:,1], P2[:,0])
    angQ = np.arctan2(Q2[:,1], Q2[:,0])
    dphi = angQ - angP
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi  # wrap

    c, s = np.cos(dphi).mean(), np.sin(dphi).mean()
    mean = np.arctan2(s, c)
    return float(np.degrees(mean))

def consecutive_frame_angles(points_seq: list[np.ndarray], axis_seq: np.ndarray | None = None):
    """
    Convenience: compute per-frame angles for a sequence of clouds.
    points_seq: list of (N,3) numpy arrays for frames 0..T
    axis_seq: optional (3,) numpy axis (constant); if given, returns also projected-angle list.
    Returns:
      kabsch_angles_deg: [T] list
      about_axis_angles_deg or None: [T] list if axis_seq provided
    """
    T = len(points_seq) - 1
    kabsch_list = []
    axis_list = [] if axis_seq is not None else None

    for t in range(1, T + 1):
        A = points_seq[t-1]
        B = points_seq[t]
        Ac = A - A.mean(axis=0, keepdims=True)
        Bc = B - B.mean(axis=0, keepdims=True)
        ang_k, _ = kabsch_rotation_angle_deg(Ac, Bc)
        kabsch_list.append(ang_k)

        if axis_seq is not None:
            ang_ax = rotation_about_given_axis_deg(Ac, Bc, axis_seq)
            axis_list.append(ang_ax)

    return kabsch_list, axis_list
