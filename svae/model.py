import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
from utils.load_data import load_dir, reshape_data
from svae.pqmf import PQMF

class WaveSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, dir, segment_length, sr=48000):
        self.waves = reshape_data(load_dir(dir, sr), segment_length)
        self.sr = sr

    def __len__(self):
        return self.waves.shape[0]
    
    def __getitem__(self, idx):
        return self.waves[idx]

class SingleVAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=8, n_channels = 1, sr = 48000, n_kernels = 64):
        super().__init__()
        self.block_length = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim 
        self.n_channels = n_channels
        self.nfft = 400
        self.sr = 48000
        self.nmels = 32
        self.hop_length = self.nfft//2
        self.n_kernels = n_kernels

        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate = self.sr, n_fft = self.nfft, n_mels = self.nmels, hop_length = self.hop_length)
        self.db = torchaudio.transforms.AmplitudeToDB()

        # encoder
        # self.wav2hid = nn.Sequential(
        #     nn.Conv1d(n_channels, h_dim//4, kernel_size=3, stride=2, padding=(3-1)//2, dilation=1),
        #     nn.ReLU(),
        #     nn.Conv1d(h_dim//4, h_dim//2, kernel_size=5, stride=2, padding=(5-1)//2, dilation=2),
        #     nn.ReLU(),
        #     nn.Conv1d(h_dim//2, h_dim, kernel_size=5, stride=2, padding=(5-1)//2, dilation=2),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1)
        # )
        self.mel2hid = nn.Linear(self.nmels*(1 + (self.block_length + 2 * (self.nfft//2) - self.nfft)//self.hop_length), h_dim)
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z2hid = nn.Linear(z_dim, n_kernels*input_dim//16) # same flattened length as flattened pre-pooled hidden output
        self.hid2wav = nn.Sequential(
            nn.ConvTranspose1d(n_kernels, n_kernels//2, kernel_size=3, stride=2, padding=(3-1)//2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_kernels//2, n_kernels//4, kernel_size=3, stride=2, padding=(3-1)//2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_kernels//4, n_kernels//8, kernel_size=3, stride=2, padding=(3-1)//2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_kernels//8, n_channels, kernel_size=3, stride=2, padding=(3-1)//2, output_padding=1)
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        mel_db = self.db(self.mel(x))

        h = self.relu(self.mel2hid(mel_db.reshape(mel_db.shape[0], -1)))
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z2hid(z))
        unpooled = h.view(z.shape[0], self.n_kernels, self.block_length//16) # same shape as pre-pooled hidden output
        x_hat = torch.tanh(self.hid2wav(unpooled))
        return x_hat

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)

        z_reparam = mu + sigma*epsilon
        x_hat = self.decode(z_reparam)

        return x_hat, mu, sigma

class PQMFVAE(nn.Module):
    """
    PQMF-to-waveform VAE model based on IRCAM's RAVE:
    https://github.com/acids-ircam/RAVE/
    """
    def __init__(self,
                 pqmf: PQMF,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [3, 3, 3, 3],
                 zdim: int = 128,
                 nchannels: int = 1,
                 strides: list[int] = [4, 4, 4, 2],
                 dilations: list[int] = [1, 3, 9]):
        super().__init__()

        self.pqmf = pqmf

        # encoder
        self.pqmf2hid = EncoderStack(pqmf.N*nchannels, nkernels, kernel_sizes, strides)
        self.hid2mu = nn.Sequential(nn.ZeroPad1d(get_padding(5)),
                                    nn.Conv1d(nkernels[-1],
                                        zdim,
                                        kernel_size=5))
        self.hid2sigma = nn.Sequential(nn.ZeroPad1d(get_padding(5)),
                                       nn.Conv1d(nkernels[-1],
                                            zdim,
                                            kernel_size=5),
                                       nn.Softplus())
        
        # decoder
        self.z2hid = DecoderStack(zdim, nkernels, kernel_sizes, strides, dilations)
        self.hid2wave = nn.Sequential(nn.ZeroPad1d(get_padding(7)),
                                      nn.Conv1d(nkernels[0],
                                                nchannels*pqmf.N,
                                                kernel_size=7),
                                      nn.BatchNorm1d(nchannels*pqmf.N))
        self.hid2loud = nn.Sequential(nn.ZeroPad1d(get_padding(3)),
                                        nn.Conv1d(nkernels[0], 1, kernel_size=3))
    
    def encode(self, x):
        x_mb = self.pqmf(x)
        h = self.pqmf2hid(x_mb)
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        h = self.z2hid(z)
        x_wav, x_loud = self.hid2wave(h), self.hid2loud(h)

        x_mb = x_wav * x_loud
        x_hat = self.pqmf.synthesis(x_mb)
        return x_hat
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)

        z_reparam = mu + sigma*epsilon
        x_hat = torch.tanh(self.decode(z_reparam))

        return x_hat, mu, sigma

class EncoderStack(nn.Module):
    def __init__(self,
                 nmels: int,
                 nkernels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int]):
            super().__init__()

            self.net = [nn.Sequential(
                nn.ZeroPad1d(get_padding(kernel_sizes[0], strides[0])),
                nn.Conv1d(nmels,
                          nkernels[0],
                          kernel_size=kernel_sizes[0],
                          stride=strides[0]),
                    nn.BatchNorm1d(nkernels[0]),
                    nn.LeakyReLU(.2))]
            for i in range(1, len(nkernels)):
                encoder_block = nn.Sequential(
                    nn.ZeroPad1d(get_padding(kernel_sizes[i], strides[i])),
                    nn.Conv1d(
                        nkernels[i-1],
                        nkernels[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i]),
                    nn.BatchNorm1d(nkernels[i]),
                    nn.LeakyReLU(.2)
                )
                self.net.append(encoder_block)
            self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)

class ResidualStack(nn.Module):
    def __init__(self,
                 nkernels: int,
                 kernel_size: int,
                 dilations: list[int]):
        super().__init__()

        residual_block = []
        for d in dilations:
            residual_layer = nn.Sequential(
                nn.ZeroPad1d(get_padding(kernel_size, dilation=d)),
                nn.LeakyReLU(.2),
                nn.Conv1d(
                    nkernels,
                    nkernels,
                    kernel_size=kernel_size,
                    dilation=d)
            )
            residual_block.append(residual_layer)
        residual_block = nn.Sequential(*residual_block)
        self.net = ResidualAdd(pre=nn.Identity(), post=residual_block)

    def forward(self, x):
        return self.net(x)

class ResidualAdd(nn.Module):
    def __init__(self, pre, post):
        super().__init__()
        self.pre = pre
        self.post = post

    def forward(self, x):
        return self.pre(x) + self.post(x)

class UpsamplingLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 stride: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=stride*2,
                stride=stride, 
                padding=stride//2)
        )
    
    def forward(self, x):
        return self.net(x)

class DecoderStack(nn.Module):
    def __init__(self,
                 zdim: int,
                 nkernels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int],
                 dilations: list[int]):
        super().__init__()

        # reverse conv params -> bottleneck structure
        strides = strides[::-1]
        nkernels = nkernels[::-1]
        kernel_sizes = kernel_sizes[::-1]

        self.net = [nn.Sequential(
            UpsamplingLayer(zdim, nkernels[0], strides[0]),
            ResidualStack(nkernels[0], kernel_sizes[0], dilations)
        )]
        for i in range(1, len(nkernels)):
            self.net.append(nn.Sequential(
                UpsamplingLayer(nkernels[i-1], nkernels[i], strides[i]),
                ResidualStack(nkernels[i], kernel_sizes[i], dilations)
            ))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, mode = "centered"):
        """
        Computes 'same' padding given a kernel size, stride an dilation.

        Copied from cached_conv by IRCAM:
        https://github.com/acids-ircam/cached_conv/blob/master/cached_conv/convs.py
        """
        if kernel_size == 1: return (0, 0)
        p = (kernel_size - 1) * dilation + 1
        half_p = p // 2
        if mode == "centered":
            p_right = p // 2
            p_left = (p - 1) // 2
        elif mode == "causal":
            p_right = 0
            p_left = p // 2 + (p - 1) // 2
        elif mode == "anticausal":
            p_right = p // 2 + (p - 1) // 2
            p_left = 0
        else:
            raise Exception(f"Padding mode {mode} is not valid")
        return (p_left, p_right)
        # return p_right

if __name__ == "__main__":
    x = torch.randn(4, 1, 49153)
    pqmf = PQMF()
    print(pqmf.synthesis(pqmf(x)).shape)
    # vae = SingleVAE(input_dim = 4800)
    vae = PQMFVAE(pqmf)
    x_hat, mu, sigma = vae(x)
    print(x_hat.shape, mu.shape, sigma.shape)