from torch.utils.data import Dataset
import torch


class CouetteDataset(Dataset):

    def __init__(self, path):

        data = torch.load(path)

        self.X = data["X"].float()
        self.Y = data["Y"].float()
        self.nu = data["nu"].float()

        self.y = data["y"].float()

        # normalization

        self.X_mean = self.X.mean()
        self.X_std = self.X.std()

        self.Y_mean = self.Y.mean()
        self.Y_std = self.Y.std()

        self.nu_mean = self.nu.mean()
        self.nu_std  = self.nu.std()

        self.X = (
            self.X - self.X_mean
        ) / self.X_std

        self.Y = (
            self.Y - self.Y_mean
        ) / self.Y_std

        self.nu = (
            self.nu - self.nu_mean
        ) / self.nu_std

        # sanity checks
        assert not torch.isnan(self.X).any()
        assert not torch.isnan(self.Y).any()
        assert not torch.isnan(self.nu).any()

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        u = self.X[idx]

        nu = self.nu[idx]

        nu_channel = torch.ones_like(
            self.y
        ) * nu

        inp = torch.stack(
            [
                u,
                self.y,
                nu_channel
            ],
            dim=0
        )

        target = self.Y[idx].unsqueeze(0)

        return {
            "a": inp,
            "u": target
        }
