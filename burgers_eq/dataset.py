from torch.utils.data import Dataset
import torch

class BurgersDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)

        self.a = data["a"].float()
        self.u = data["u"].float()

        self.a_mean = self.a.mean()
        self.a_std  = self.a.std()

        self.u_mean = self.u.mean()
        self.u_std  = self.u.std()

        self.a = (self.a - self.a_mean) / self.a_std
        self.u = (self.u - self.u_mean) / self.u_std

        self.N = self.a.shape[-1]
        self.grid = torch.linspace(0, 1, self.N)

        self.stats = {
            "a_mean": self.a_mean,
            "a_std": self.a_std,
            "u_mean": self.u_mean,
            "u_std": self.u_std
        } 

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        a = self.a[idx]

        inp = torch.stack(
            [a, self.grid],
            dim=0
        )

        return {
            "a": inp,
            "u": self.u[idx].unsqueeze(0)
        }
