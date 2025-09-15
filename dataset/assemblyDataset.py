# dataset/assemblyDataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import config.config as cfg

class GearAssemblyDataset(Dataset):
    def __init__(self, base_path, list_file):
        with open(list_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines()]

        self.file_paths = [
            os.path.join(base_path, fname)
            for fname in self.file_names
            if fname.endswith(".npz") and os.path.exists(os.path.join(base_path, fname))
        ]

        assert self.file_paths, f"No valid .npz files found in {list_file}"

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])

        # Load full tensors
        point_cloud = torch.tensor(data["point_cloud"], dtype=torch.float32)         # (P, N, 3)

        # === PATCH: Robust mask loader ===
        mask_np = data["mask"]

        if np.isscalar(mask_np):  # scalar float32
            mask = torch.tensor([bool(mask_np)], dtype=torch.bool)
        elif isinstance(mask_np, np.ndarray):  # array case
            mask = torch.tensor(mask_np.astype(bool), dtype=torch.bool)
        else:
            raise TypeError(f"Unexpected mask type: {type(mask_np)}")                                    # array case

        motion_type = torch.tensor(data["motion_type"], dtype=torch.long)            # (P,)
        coupling_matrix = torch.tensor(data["coupling_matrix"], dtype=torch.float32) # (P, P)
        center = torch.tensor(data["center"], dtype=torch.float32)                   # (P, 3)
        axis = torch.tensor(data["axis"], dtype=torch.float32)                       # (P, 3)
        transform = torch.tensor(data["transform"], dtype=torch.float32)             # (T, P, 4, 4)
        lav = torch.tensor(data["lav"], dtype=torch.float32)                         # (T-1, P, 6)

        scaler_factor = float(data["scaler_factor"])
        scaler_center = torch.tensor(data["scaler_center"], dtype=torch.float32)

        # === Trim ===
        P = cfg.NUM_PARTS
        N = cfg.NUM_POINTS
        T = cfg.NUM_FRAMES

        point_cloud = point_cloud[:P, :N, :]        # (P, N, 3)
        mask = mask[:P]                             # (P,)
        motion_type = motion_type[:P]
        coupling_matrix = coupling_matrix[:P, :P]
        center = center[:P]
        axis = axis[:P]
        lav = lav[:T, :P]                           # (T, P, 6)
        transform = transform[:T+1, :P]             # (T+1, P, 4, 4)

        return {
            "point_cloud": point_cloud,
            "mask": mask,
            "motion_type": motion_type,
            "coupling_matrix": coupling_matrix,
            "center": center,
            "axis": axis,
            "transform": transform,
            "lav": lav,
            "scaler_factor": scaler_factor,
            "scaler_center": scaler_center
        }
