import torch
import torch.nn as nn
from core.spectral_conv import SpectralConv1d, SpectralConv2d, SpectralConv3d

class FNO1d(nn.Module):
    def __init__(
            self,
            modes,
            width,
            in_channels=2,
            out_channels=1,
            n_layers=4,
            act_fnct = nn.GELU()
            ):
        super().__init__()

        self.lift = nn.Conv1d(in_channels, width, 1)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "spectral": SpectralConv1d(width, width, modes),
                        "local": nn.Conv1d(width, width, 1)
                    }
                )
            )

        self.project = nn.Sequential(
            nn.Conv1d(width, 128, 1),
            act_fnct,
            nn.Conv1d(128, out_channels, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers[:-1]:
            x = layer["spectral"](x) + layer['local'](x)
            x = torch.nn.functional.gelu(x)

        # last layer no activaation unbounded output
        layer = self.layers[-1]
        x = layer['spectral'](x) + layer['local'](x)
        return self.project(x)
        
# x = torch.randn(4, 2, 1024)
# model = FNO1d(modes=16, width=64)
# y = model(x)

# print(y.shape)



class FNO2d(nn.Module):
    def __init__(
            self,
            modesx,
            modesy,
            width,
            in_channels=2,
            out_channels=1,
            n_layers=4,
            act_fnct = nn.GELU()
        ):
        super().__init__()

        self.lift = nn.Conv2d(in_channels, width, 1)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "spectral": SpectralConv2d(width, width, modesx, modesy),
                "local": nn.Conv2d(width, width, 1)
            }))

        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            act_fnct,
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        for layer in self.layers[:-1]:
            x = layer["spectral"](x) + layer['local'](x)
            x = torch.nn.functional.gelu(x)

        # last layer no activaation unbounded output
        layer = self.layers[-1]
        x = layer['spectral'](x) + layer['local'](x)
        return self.project(x)
    


class FNO3d(nn.Module):
    def __init__(
        self,
        modesx: int,
        modesy: int,
        modesz: int,
        width: int,
        in_channels: int = 3,
        out_channels: int = 1,
        n_layers: int = 4,
        act_fnct: nn.Module = nn.GELU(),
    ):
        super().__init__()

        self.activation = act_fnct

        self.lift = nn.Conv3d(in_channels, width, 1)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "spectral": SpectralConv3d(width, width, modesx, modesy, modesz),
                        "local": nn.Conv3d(width, width, 1)
                    }
                )
            )

        self.project = nn.Sequential(
            nn.Conv3d(width, 128, 1),
            act_fnct,
            nn.Conv3d(128, out_channels, 1)
        )

    def forward(self, x):
        x = self.lift(x)

        for layer in self.layers[:-1]:
            x = layer["spectral"](x) + layer['local'](x)
            x = self.activation(x)

        # last layer no activation unbounded output
        layer = self.layers[-1]
        x = layer['spectral'](x) + layer['local'](x)
        return self.project(x)
        
if __name__ == "__main__":

    x = torch.randn(
        4,      # batch
        3,      # channels
        64,     # Nx
        64,     # Ny
        64      # Nz

    )

    model = FNO3d(
        modesx=12,
        modesy=12,
        modesz=12,
        width=32,
        in_channels=3,
        out_channels=1
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
