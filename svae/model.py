import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
from utils.load_data import load_dir, reshape_data

class WaveSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, dir, segment_length, sr=48000):
        self.waves = reshape_data(load_dir(dir, sr), segment_length)
        self.sr = sr

    def __len__(self):
        return self.waves.shape[0]
    
    def __getitem__(self, idx):
        return self.waves[idx]

class SingleVAE(nn.Module):
    def __init__(self, input_dim, h_dim=64, z_dim=8, kernel_size=3, n_channels = 1):
        super().__init__()
        self.block_length = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim 
        self.n_channels = n_channels

        # encoder
        self.wav2hid = nn.Sequential(
            nn.Conv1d(n_channels, h_dim//4, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv1d(h_dim//4, h_dim//2, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv1d(h_dim//2, h_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z2hid = nn.Linear(z_dim, h_dim*input_dim//8) # same flattened length as flattened pre-pooled hidden output
        self.hid2wav = nn.Sequential(
            nn.ConvTranspose1d(h_dim, h_dim//2, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim//2, h_dim//4, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim//4, n_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.wav2hid(x).squeeze(-1)
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z2hid(z))
        unpooled = h.view(z.shape[0], self.h_dim, self.block_length//8) # same shape as pre-pooled hidden output
        x_hat = torch.tanh(self.hid2wav(unpooled))
        return x_hat

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)

        z_reparam = mu + sigma*epsilon
        x_hat = self.decode(z_reparam)

        return x_hat, mu, sigma

if __name__ == "__main__":
    x = torch.randn(1, 1, 4800)
    vae = SingleVAE(input_dim = 4800)

    x_hat, mu, sigma = vae(x)
    print(x_hat.shape, mu.shape, sigma.shape)