import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    """ 1D spectral convolution : FFT -> learnable multiply -> IFFT . """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes
        ):

        super().__init__()
        self.in_channels= in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / ( in_channels * out_channels)

        # learnable complex weights: shape(in_c, out_c, modes)
        self.weights = nn.Parameter(
            scale 
            * torch.rand(
                in_channels,
                out_channels,
                modes,
                dtype= torch.cfloat
            )
        )
        
    def complex_mul1d(self, input, weights):
        # input : ( batch , in_c , modes )
        # weights : ( in_c , out_c , modes )
        return torch.einsum('bix, iox -> box', input, weights)
    
    def forward(self, x):
        # x = (batch, in_channels, N)

        batch, _ , N = x.shape

        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batch,
            self.out_channels,
            N // 2 +1,
            device= x.device, 
            dtype= torch.cfloat
        )
        modes = min(self.modes, N // 2 + 1)

        out_ft[:, :, :modes] = self.complex_mul1d(
            x_ft[:, :, :modes],
            self.weights
            )
        x_out = torch.fft.irfft(
            out_ft, 
            n = N,
            norm = "forward"
        )
        return x_out


class SpectralConv2d(nn.Module):
    """ 2D spectral convolution : FFT -> learnable multiply -> IFFT . """

    def __init__(self, in_channels, out_channels, modesx, modesy):
        super().__init__()
        self.in_channels= in_channels
        self.out_channels = out_channels
        self.modesx = modesx
        self.modesy = modesy

        scale = 1.0 / ( in_channels * out_channels)

        # learnable complex weights: shape(in_c, out_c, modesx, modesy)
        self.w1 = nn.Parameter(
            scale 
            * torch.rand(
                in_channels,
                out_channels,
                modesx,
                modesy,
                dtype= torch.cfloat
            )
        )
        self.w2 = nn.Parameter(
            scale 
            * torch.rand(
                in_channels,
                out_channels,
                modesx,
                modesy,
                dtype= torch.cfloat
            )
        )
        
    def compl_mul2d(self, input, weights):
        # input : ( batch , in_c , modesx, modesy )
        # weights : ( in_c , out_c , modesx, modesy )
        return torch.einsum('bixy, ioxy -> boxy', input, weights)
    
    def forward(self, x):
        # x = (batch, in_channels, N)
        batch, _ , Nx, Ny = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            Nx,
            Ny // 2 +1,
            device= x.device, 
            dtype= torch.cfloat
        )
        
        out_ft[:, :, :self.modesx, :self.modesy] = self.compl_mul2d(
            x_ft[:, :, :self.modesx, :self.modesy],
            self.w1)
        x_out = torch.fft.irfft2(out_ft, s = (Nx, Ny))
        return x_out
    

import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    """
    3D Fourier Layer

    FFT -> Spectral Multiplication -> IFFT

    Input:
        (batch, in_channels, Nx, Ny, Nz)

    Output:
        (batch, out_channels, Nx, Ny, Nz)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        modes_z: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        scale = 1.0 / (in_channels * out_channels)

        # (+x,+y,+z)
        self.weights1 = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                modes_z,
                dtype=torch.cfloat,
            )
        )

        # (-x,+y,+z)
        self.weights2 = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                modes_z,
                dtype=torch.cfloat,
            )
        )

        # (+x,-y,+z)
        self.weights3 = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                modes_z,
                dtype=torch.cfloat,
            )
        )

        # (-x,-y,+z)
        self.weights4 = nn.Parameter(
            scale * torch.rand(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                modes_z,
                dtype=torch.cfloat,
            )
        )

    def complex_mul3d(self, x, weights):
        """
        x:
            (batch, in_channels, mx, my, mz)

        weights:
            (in_channels, out_channels, mx, my, mz)

        returns:
            (batch, out_channels, mx, my, mz)
        """
        return torch.einsum(
            "bixyz,ioxyz->boxyz",
            x,
            weights,
        )

    def forward(self, x):

        batch_size, _, Nx, Ny, Nz = x.shape

        x_ft = torch.fft.rfftn(
            x,
            dim=(-3, -2, -1),
        )

        mx = min(self.modes_x, Nx)
        my = min(self.modes_y, Ny)
        mz = min(self.modes_z, Nz // 2 + 1)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            Nx,
            Ny,
            Nz // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # (+x,+y,+z)

        out_ft[:, :, :mx, :my, :mz] = self.complex_mul3d(
            x_ft[:, :, :mx, :my, :mz],
            self.weights1[:, :, :mx, :my, :mz],
        )

        # (-x,+y,+z)

        out_ft[:, :, -mx:, :my, :mz] = self.complex_mul3d(
            x_ft[:, :, -mx:, :my, :mz],
            self.weights2[:, :, :mx, :my, :mz],
        )

        # (+x,-y,+z)

        out_ft[:, :, :mx, -my:, :mz] = self.complex_mul3d(
            x_ft[:, :, :mx, -my:, :mz],
            self.weights3[:, :, :mx, :my, :mz],
        )

        # (-x,-y,+z)

        out_ft[:, :, -mx:, -my:, :mz] = self.complex_mul3d(
            x_ft[:, :, -mx:, -my:, :mz],
            self.weights4[:, :, :mx, :my, :mz],
        )

        x = torch.fft.irfftn(
            out_ft,
            s=(Nx, Ny, Nz),
            dim=(-3, -2, -1),
        )

        return x

if __name__ == "__main__":
    x = torch.randn(
        4,      # batch
        7,      # channels
        64,     # Nx
        64,     # Ny
        64      # Nz
    )

    model = SpectralConv3d(
        in_channels=7,
        out_channels=3,
        modes_x=12,
        modes_y=12,
        modes_z=12
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
