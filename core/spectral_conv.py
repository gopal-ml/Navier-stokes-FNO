import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    " " " 1 D spectral convolution : FFT -> learnable multiply -> IFFT . " " "

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels= in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / ( in_channels * out_channels)

        # learnable complex weights: shape(in_c, out_c, modes)
        self.weights = nn.Parameter(scale * torch.rand(in_channels,
                                                       out_channels,
                                                       modes,
                                                       dtype= torch.cfloat))
        
    def compl_mul1d(self, input, weights):
        # input : ( batch , in_c , modes )
        # weights : ( in_c , out_c , modes )
        return torch.einsum('bix, iox -> box', input, weights)
    
    def forward(self, x):
        # x = (batch, in_channels, N)
        batch, _ , N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batch,
                                self.out_channels,
                                N // 2 +1,
                                device= x.device, 
                                dtype= torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes],
                                                        self.weights)
        x_out = torch.fft.irfft(out_ft, n = N)
        return x_out


class SpectralConv2d(nn.Module):
    " " " 2 D spectral convolution : FFT -> learnable multiply -> IFFT . " " "

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
        # input : ( batch , in_c , modes )
        # weights : ( in_c , out_c , modes )
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

# x = torch.randn(
#     4,
#     32,
#     64,
#     64
# )

# layer = SpectralConv2d(
#     32,
#     32,
#     modesx=12,
#     modesy=12
# )

# y = layer(x)

# print(y.shape)
