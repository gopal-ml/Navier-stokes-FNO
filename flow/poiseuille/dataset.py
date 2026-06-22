from torch.utils.data import Dataset
import torch


class PoiseuilleDataset(Dataset):

    def __init__(self, path):

        data = torch.load(path)

        self.X = data["X"].float()
        self.Y = data["Y"].float()
        self.nu = data["nu"].float()
        self.G = data["G"].float()

        self.y = data["y"].float()
        
        # sanity checks
        assert self.X.shape == self.Y.shape

        assert len(self.X) == len(self.nu)
        assert len(self.X) == len(self.G)

        # normalization

        self.X_mean = self.X.mean()
        self.X_std = self.X.std() + 1e-8

        self.Y_mean = self.Y.mean()
        self.Y_std = self.Y.std() + 1e-8

        self.nu_mean = self.nu.mean()
        self.nu_std  = self.nu.std() + 1e-8

        self.G_mean = self.G.mean()
        self.G_std = self.G.std() + 1e-8

        self.X = (
            self.X - self.X_mean
        ) / self.X_std

        self.Y = (
            self.Y - self.Y_mean
        ) / self.Y_std

        self.nu = (
            self.nu - self.nu_mean
        ) / self.nu_std

        self.G = (
            self.G - self.G_mean
        ) / self.G_std

        # sanity checks
        assert not torch.isnan(self.X).any()
        assert not torch.isnan(self.Y).any()
        assert not torch.isnan(self.nu).any()
        assert not torch.isnan(self.G).any()

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        u = self.X[idx]
        nu = self.nu[idx]
        G = self.G[idx]

        nu_channel = torch.ones_like(
            self.y
        ) * nu
        G_channel = torch.ones_like(
            self.y
        ) * G

        inp = torch.stack(
            [
                u,
                self.y,
                nu_channel,
                G_channel
            ],
            dim=0
        )

        target = self.Y[idx].unsqueeze(0)

        return {
            "a": inp,
            "u": target
        }
