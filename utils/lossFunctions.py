import torch
from liegroups.torch import SE3
from liegroups.torch import SO3

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

def compute_geodesic_rot_loss(pred_rot, gt_rot, mask):
    B, P, T, _ = pred_rot.shape
    pred_rot_flat = pred_rot.reshape(-1, 3)
    gt_rot_flat = gt_rot.reshape(-1, 3)

    R_pred = SO3.exp(pred_rot_flat).as_matrix()  # (B*P*T, 3, 3)
    R_gt = SO3.exp(gt_rot_flat).as_matrix()

    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
    cos_angle = (R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2] - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cos_angle)  # (B*P*T,)

    angles = angles.view(B, P, T)
    mask_exp = mask.unsqueeze(-1).expand(-1, -1, T)
    return (angles * mask_exp).sum() / mask_exp.sum()

def se3_exp_map(lav):  # (P, 6)
    omega = lav[:, :3]  # rotation
    v = lav[:, 3:]      # translation

    theta = torch.norm(omega, dim=1, keepdim=True) + 1e-8  # avoid zero-div
    axis = omega / theta

    # Rodrigues' formula for rotation
    K = hat(axis)
    I = torch.eye(3, device=lav.device).unsqueeze(0)
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)

    # Approximate left Jacobian J for translation (safe for small theta)
    J = I + (1 - torch.cos(theta)).unsqueeze(-1) * K + (theta - torch.sin(theta)).unsqueeze(-1) * torch.bmm(K, K)
    t = torch.bmm(J, v.unsqueeze(-1)).squeeze(-1)

    # Form SE(3)
    T = torch.zeros(lav.shape[0], 4, 4, device=lav.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T  # (P, 4, 4)

def hat(vec):  # vec: (B, 3)
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)  # (B, 3, 3)


def lav_to_se3_sequence(lav):  # lav: (P, T, 6)
    P, T, _ = lav.shape
    traj = []

    identity = torch.eye(4, device=lav.device).unsqueeze(0).repeat(P, 1, 1)  # (P, 4, 4)
    traj.append(identity)

    lav_split = lav.unbind(dim=1)  # List of (P, 6) tensors

    for lav_t in lav_split:
        lav_t = lav_t.contiguous()  # ensure clean memory
        dT = se3_exp_map(lav_t)
        traj.append(torch.bmm(traj[-1], dT))

    return torch.stack(traj, dim=1)  # (P, T+1, 4, 4)




def apply_transform_to_pc(pc, se3_seq):  # pc: (B, P, 3, N), se3_seq: (B, P, T+1, 4, 4)
    B, P, _, N = pc.shape
    T = se3_seq.shape[2] - 1  # total frames = T+1

    # Convert pc to homogeneous coords: (B, P, 4, N)
    ones = torch.ones((B, P, 1, N), device=pc.device)
    homo_pc = torch.cat([pc, ones], dim=2)  # (B, P, 4, N)

    # Expand over time
    homo_pc = homo_pc.unsqueeze(2).repeat(1, 1, T + 1, 1, 1)  # (B, P, T+1, 4, N)

    # Reshape for batch matmul
    homo_pc = homo_pc.permute(0, 1, 2, 4, 3).reshape(B * P * (T + 1), N, 4)  # (BPT, N, 4)
    se3_seq = se3_seq.reshape(B * P * (T + 1), 4, 4)  # (BPT, 4, 4)

    # Apply batched matmul
    transformed = torch.bmm(homo_pc, se3_seq.transpose(1, 2))  # (BPT, N, 4)

    # Reshape back to (B, P, T+1, 3, N)
    transformed = transformed[:, :, :3]  # drop homogeneous coord
    transformed = transformed.reshape(B, P, T + 1, N, 3).permute(0, 1, 2, 4, 3)  # (B, P, T+1, 3, N)

    return transformed  # shape: (B, P, T+1, 3, N)

def pairwise_distances(x, y):
    x_sq = x.pow(2).sum(dim=-2, keepdim=True)
    y_sq = y.pow(2).sum(dim=-2, keepdim=True)
    inner = torch.matmul(x.transpose(-1, -2), y)
    dist = x_sq - 2 * inner + y_sq.transpose(-1, -2)
    return torch.clamp(dist, min=1e-6)


def compute_losses(
    lav_pred, lav_gt, pc, mask,
    loss_config,
):
    """
    Computes different combinations of loss terms depending on config.

    Args:
        lav_pred: (B, P, T, 6)
        lav_gt:   (B, P, T, 6)
        pc:       (B, P, 3, N) input point cloud
        mask:     (B, P)
        loss_config: dict with booleans and weights:
            {
                'use_l2': True,
                'use_l2_first_frame_only': False,
                'use_lav_const': True,
                'use_chamfer': False,
                'lambda_trans': 1.0,
                'lambda_rot': 1.0,
                'lambda_lav_const': 1.0,
                'lambda_chamfer': 1.0
            }

    Returns:
        total_loss, loss_dict (individual loss values)
    """
    B, P, T, _ = lav_pred.shape
    losses = {}
    total = 0.0

    # --- L2 Loss ---
    if loss_config.get("use_l2", False):
        pred_trans, pred_rot = lav_pred[..., :3], lav_pred[..., 3:]
        gt_trans, gt_rot = lav_gt[..., :3], lav_gt[..., 3:]

        trans_loss = l2_loss(pred_trans, gt_trans, mask)
        losses["l2_trans"] = trans_loss
        total += loss_config["lambda_trans"] * trans_loss

        # --- Rotation uses either geodesic or L2 ---
        if loss_config.get("use_geodesic_rot", False):
            rot_loss = compute_geodesic_rot_loss(pred_rot, gt_rot, mask)
            losses["geodesic_rot"] = rot_loss
        else:
            if loss_config.get("use_l2_first_frame_only", False):
                rot_loss = l2_loss(pred_rot[:, :, 0], gt_rot[:, :, 0], mask)
            else:
                rot_loss = l2_loss(pred_rot, gt_rot, mask)
            losses["l2_rot"] = rot_loss

        total += loss_config["lambda_rot"] * rot_loss

    # --- LAV Consistency Loss ---
    if loss_config.get("use_lav_const", False):
        lav_diff = lav_pred[:, :, 1:] - lav_pred[:, :, :-1]
        mean_diff = lav_diff.mean(dim=2, keepdim=True)
        lav_const = (lav_diff - mean_diff).pow(2).mean()
        losses["lav_const"] = lav_const
        total += loss_config["lambda_lav_const"] * lav_const


    return total, losses
