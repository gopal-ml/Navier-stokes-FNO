# ============================================================
# IMPORTS
# ============================================================

import torch
from torch.utils.data import Dataset


# ============================================================
# Cavity3D Dataset
# ============================================================

class Cavity3DDataset(Dataset):
    """
    Dataset for 3D lid-driven cavity flow.
    Input channels
    --------------
    0 : u
    1 : v
    2 : w
    3 : nu
    4 : x
    5 : y
    6 : z
    Output channels
    ---------------
    0 : u_next
    1 : v_next
    2 : w_next
    """

    def __init__(
        self,
        path,
        normalize=True,
    ):

        data = torch.load(
            path,
            map_location="cpu",
        )

        self.X = data["X"].float()
        self.Y = data["Y"].float()

        self.x = data["x"]
        self.y = data["y"]
        self.z = data["z"]
        
        # One viscosity per sample
        self.sample_nu = data["sample_nu"]

        self.input_channels = data["input_channels"]
        self.output_channels = data["output_channels"]

        self.normalize = normalize

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------
        dynamic = slice(0,3)

        self.X_mean = self.X[:,dynamic].mean(
            dim=(0,2,3,4),
            keepdim=True,
        )

        self.X_std = self.X[:,dynamic].std(
            dim=(0,2,3,4),
            keepdim=True,
        )

        self.Y_mean = self.Y.mean(
            dim=(0,2,3,4),
            keepdim=True,
        )

        self.Y_std = self.Y.std(
            dim=(0,2,3,4),
            keepdim=True,
        )

        self.X_std[self.X_std < 1e-8] = 1.0
        self.Y_std[self.Y_std < 1e-8] = 1.0

        if normalize:

            self.X[:,dynamic] = (
                self.X[:,dynamic]-self.X_mean
            )/self.X_std

            self.Y = (
                self.Y - self.Y_mean
            ) / self.Y_std

    # ========================================================

    def __len__(self):

        return len(self.X)

    # ========================================================

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]

    # ========================================================

    def denormalize_output(self, y):

        return y * self.Y_std + self.Y_mean
