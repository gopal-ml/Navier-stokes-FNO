from torch.utils.data import Dataset
import torch


class CavityDrivenDataset(Dataset):

    def __init__(self, path):

        data = torch.load(path)

        # ==================================
        # LOAD
        # ==================================

        self.X_u = data["X_u"].float()
        self.X_v = data["X_v"].float()

        self.Y_u = data["Y_u"].float()
        self.Y_v = data["Y_v"].float()

        self.nu = data["nu"].float()

        self.x = data["x"].float()
        self.y = data["y"].float()

        # ==================================
        # NORMALIZATION STATS
        # ==================================

        self.u_mean = self.X_u.mean()
        self.u_std = self.X_u.std()

        self.v_mean = self.X_v.mean()
        self.v_std = self.X_v.std()

        self.nu_mean = self.nu.mean()
        self.nu_std = self.nu.std()

        # ==================================
        # NORMALIZE INPUTS
        # ==================================

        self.X_u = (
            self.X_u - self.u_mean
        ) / self.u_std

        self.X_v = (
            self.X_v - self.v_mean
        ) / self.v_std

        # ==================================
        # NORMALIZE TARGETS
        # ==================================

        self.Y_u = (
            self.Y_u - self.u_mean
        ) / self.u_std

        self.Y_v = (
            self.Y_v - self.v_mean
        ) / self.v_std

        # ==================================
        # NORMALIZE NU
        # ==================================

        self.nu = (
            self.nu - self.nu_mean
        ) / self.nu_std

        # ==================================
        # 2D GRID
        # ==================================

        self.X_grid, self.Y_grid = torch.meshgrid(
            self.x,
            self.y,
            indexing="xy"
        )

        # ==================================
        # SANITY CHECKS
        # ==================================

        assert not torch.isnan(self.X_u).any()
        assert not torch.isnan(self.X_v).any()

        assert not torch.isnan(self.Y_u).any()
        assert not torch.isnan(self.Y_v).any()

        assert not torch.isnan(self.nu).any()

    def __len__(self):

        return len(self.X_u)

    def __getitem__(self, idx):

        u = self.X_u[idx]
        v = self.X_v[idx]

        nu = self.nu[idx]

        nu_channel = torch.ones_like(
            self.X_grid
        ) * nu

        inp = torch.stack(
            [
                u,
                v,
                self.X_grid,
                self.Y_grid,
                nu_channel
            ],
            dim=0
        )

        target = torch.stack(
            [
                self.Y_u[idx],
                self.Y_v[idx]
            ],
            dim=0
        )

        return {
            "a": inp,
            "u": target
        }
