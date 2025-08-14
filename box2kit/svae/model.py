import torch
import numpy as np
import math
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torchaudio
from box2kit.utils.load_data import load_dir, reshape_data, load_mono
from box2kit.svae.rave_pqmf import PQMF
from audiotools import AudioSignal
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from box2kit.gantransfer.ganmodel import DiscriminatorV2
import os

class WaveSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, dir, segment_length, sr=48000):
        self.waves = reshape_data(load_dir(dir, sr), segment_length)
        self.sr = sr

    def __len__(self):
        return self.waves.shape[0]
    
    def __getitem__(self, idx):
        return self.waves[idx]

class PairedWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, query_dir, target_dir, segment_length, sr=48000):
        # if query_data.shape != target_data.shape:
        #     raise Exception(f"Query dataset and target dataset must have the same size (query dataset has shape {query_data.shape}, while target dataset has shape {target_data.shape})")
        # self.query_data = query_data
        # self.target_data = target_data
        self.query_data = reshape_data(load_dir(query_dir, sr), segment_length)
        self.target_data = reshape_data(load_dir(target_dir, sr), segment_length)

        if self.query_data.shape != self.target_data.shape:
            raise Exception(f"Query dataset and target dataset must have the same size (query dataset has shape {self.query_data.shape}, while target dataset has shape {self.target_data.shape})")
    
    def __len__(self):
        return self.query_data.shape[0]
    
    def __getitem__(self, idx: int):
        x = self.query_data[idx]
        y = self.target_data[idx]
        return x, y

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
            LinterConvTranspose1D(n_kernels, n_kernels//2, kernel_size=3, stride=2, padding=(3-1)//2),
            nn.ReLU(),
            LinterConvTranspose1D(n_kernels//2, n_kernels//4, kernel_size=3, stride=2, padding=(3-1)//2),
            nn.ReLU(),
            LinterConvTranspose1D(n_kernels//4, n_kernels//8, kernel_size=3, stride=2, padding=(3-1)//2),
            nn.ReLU(),
            LinterConvTranspose1D(n_kernels//8, n_channels, kernel_size=3, stride=2, padding=(3-1)//2)
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

class LightningVAE(pl.LightningModule):
    def __init__(self,
                 block_length: int,
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 48000,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [3, 3, 3, 3],
                 zdim: int = 128,
                 nchannels: int = 1,
                 strides: list[int] = [4, 4, 4, 2],
                 dilations: list[int] = [1, 3, 9],
                 nmog: int = 0,
                 mel_loss_fn = MelSpectrogramLoss(window_lengths=[4096, 2048, 1024, 512, 256], n_mels = [320, 160, 80, 40, 20], mel_fmin=[0,0,0,0,0], mel_fmax=[None,None,None,None,None]),
                 full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[1024, 512, 256, 128, 64, 32]),
                 mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16]),
                 lr = 1e-4):
        super().__init__()
        
        self.model = PQMFVAE(pqmf,
                             sample_rate,
                             nkernels,
                             kernel_sizes,
                             zdim,
                             nchannels,
                             strides,
                             dilations,
                             nmog)

        self.lr = lr
        self.mel_loss_fn = mel_loss_fn
        self.full_stft_loss_fn = full_stft_loss_fn
        self.mb_stft_loss_fn = mb_stft_loss_fn
        self.block_length = block_length

        self.save_hyperparameters(ignore=["pqmf", "full_stft_loss_fn", "mb_stft_loss_fn", "mel_loss_fn"])

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # forward

        x_hat, mu, sigma = self.model(batch)
        x_hat = x_hat[:,:,:batch.shape[2]] # the model works on frames of length (strides x pqmf_bands) = 4x4x4x2x16 = 2048 samples

        # losses
        x_hat_AS, x_AS = AudioSignal(x_hat, self.model.sr), AudioSignal(batch, self.model.sr)
        fullband_reconstruction_loss = self.mel_loss_fn(x_hat_AS, x_AS) + self.full_stft_loss_fn(x_hat_AS, x_AS)

        multiband_reconstruction_loss = self.mb_stft_loss_fn(AudioSignal(self.model.pqmf(x_hat), self.model.sr), AudioSignal(self.model.pqmf(batch), self.model.sr))
        reconstruction_loss = fullband_reconstruction_loss + multiband_reconstruction_loss

        kl_div = -torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # backprop
        beta = cos_annealed_beta(self.trainer.current_epoch, self.trainer.max_epochs)
        loss = reconstruction_loss + beta*kl_div

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("recon_loss", reconstruction_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("kld", kl_div, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("beta", beta, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return [optimizer], []
    
class TransferVAE(LightningVAE):
    def __init__(self,
                 block_length: int,
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 48000,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [3, 3, 3, 3],
                 zdim: int = 128,
                 nchannels: int = 1,
                 strides: list[int] = [4, 4, 4, 2],
                 dilations: list[int] = [1, 3, 9],
                 nmog: int = 0,
                 mel_loss_fn = MelSpectrogramLoss(window_lengths=[1024], n_mels = [128], mel_fmin=[0], mel_fmax=[None]),
                 full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[2048, 1024, 512, 256, 128]),
                 mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16]),
                 lr = 1e-4):
        super().__init__(block_length,
                         pqmf,
                         sample_rate,
                         nkernels,
                         kernel_sizes,
                         zdim,
                         nchannels,
                         strides,
                         dilations,
                         nmog,
                         mel_loss_fn,
                         full_stft_loss_fn,
                         mb_stft_loss_fn,
                         lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # forward

        y_hat, mu, sigma = self.model(x)
        y_hat = y_hat[:,:,:y.shape[2]] # the model works on frames of length (strides x pqmf_bands) = 4x4x4x2x16 = 2048 samples

        # losses
        y_hat_AS, y_AS = AudioSignal(y_hat, self.model.sr), AudioSignal(y, self.model.sr)
        fullband_reconstruction_loss = self.mel_loss_fn(y_hat_AS, y_AS) + self.full_stft_loss_fn(y_hat_AS, y_AS)

        multiband_reconstruction_loss = self.mb_stft_loss_fn(AudioSignal(self.model.pqmf(y_hat), self.model.sr), AudioSignal(self.model.pqmf(y), self.model.sr))
        reconstruction_loss = fullband_reconstruction_loss + multiband_reconstruction_loss

        kl_div = self.model.prior(mu, sigma)

        # backprop
        beta = cos_annealed_beta(self.trainer.current_epoch, self.trainer.max_epochs)
        loss = reconstruction_loss + beta*kl_div

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("recon_loss", reconstruction_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("kld", kl_div, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("beta", beta, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

class TransferGAN(LightningVAE):
    def __init__(self,
                 block_length: int,
                 discriminator_dims: list[int],
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 48000,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [3, 3, 3, 3],
                 zdim: int = 128,
                 nchannels: int = 1,
                 strides: list[int] = [4, 4, 4, 2],
                 dilations: list[int] = [1, 3, 9],
                 nmog: int = 0,
                 mel_loss_fn = MelSpectrogramLoss(window_lengths=[1024], n_mels = [128], mel_fmin=[0], mel_fmax=[None]),
                 full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[2048, 1024, 512, 256, 128]),
                 mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16]),
                 lr: float = 1e-4,
                 lambda_adversarial: float = 1,
                 warmup: int = 250
                 ):
        self.lambda_adversarial = lambda_adversarial
        super().__init__(block_length,
                         pqmf,
                         sample_rate,
                         nkernels,
                         kernel_sizes,
                         zdim,
                         nchannels,
                         strides,
                         dilations,
                         nmog,
                         mel_loss_fn,
                         full_stft_loss_fn,
                         mb_stft_loss_fn,
                         lr)
        
        self.discriminator = DiscriminatorV2(discriminator_dims)
        self.adversarial_phase = False
        self.real_label = 1
        self.fake_label = 0
        self.adversarial_loss_fn = torch.nn.BCELoss()
        self.warmup = warmup

        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        gen_optimizer, discr_optimizer = self.optimizers()

        # train generator
        self.toggle_optimizer(gen_optimizer)

        y_hat, mu, sigma = self.model(x)
        y_hat = y_hat[:,:,:y.shape[2]] # the model works on frames of length (strides x pqmf_bands) = 4x4x4x2x16 = 2048 samples

        # generator losses
        y_hat_AS, y_AS = AudioSignal(y_hat, self.model.sr), AudioSignal(y, self.model.sr)
        fullband_reconstruction_loss = self.mel_loss_fn(y_hat_AS, y_AS) + self.full_stft_loss_fn(y_hat_AS, y_AS)

        multiband_reconstruction_loss = self.mb_stft_loss_fn(AudioSignal(self.model.pqmf(y_hat), self.model.sr), AudioSignal(self.model.pqmf(y), self.model.sr))
        reconstruction_loss = fullband_reconstruction_loss + multiband_reconstruction_loss

        kl_div = torch.mean(self.model.prior(mu, sigma), 0)
        print(kl_div.shape)

        if self.adversarial_phase:
            stft_gen = torch.stft(y_hat.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=y_hat.device), return_complex=True).abs()
            gen_score = self.discriminator(stft_gen)
            real_labels = torch.full_like(gen_score, fill_value=self.real_label)
            # how convinced the discriminator is that generated waveforms are real
            adversarial_loss = self.adversarial_loss_fn(gen_score, real_labels)

        else:
            adversarial_loss = 0

        # backprop
        beta = cos_annealed_beta(self.trainer.current_epoch, self.trainer.max_epochs)
        gen_loss = reconstruction_loss + beta*kl_div + self.lambda_adversarial * adversarial_loss

        gen_optimizer.zero_grad()
        self.manual_backward(gen_loss)
        gen_optimizer.step()
        self.untoggle_optimizer(gen_optimizer)

        # train discriminator
        if self.adversarial_phase:
            self.toggle_optimizer(discr_optimizer)

            stft_target = stft_target = torch.stft(y.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=y.device), return_complex=True).abs()
            discr_optimizer.zero_grad()
            # how convinced the discriminator is that target waveforms are real, and generated waveforms are fake
            real_score = self.discriminator(stft_target)
            gen_score = self.discriminator(stft_gen.detach())

            real_labels = torch.full_like(real_score, fill_value=self.real_label)
            fake_labels = torch.full_like(gen_score, fill_value=self.fake_label)

            discr_loss = self.adversarial_loss_fn(real_score, real_labels) + self.adversarial_loss_fn(gen_score, fake_labels)
            self.manual_backward(discr_loss)
            discr_optimizer.step()
            self.untoggle_optimizer(discr_optimizer)
        else:
            discr_loss = 0

        self.log("d_loss", discr_loss, prog_bar=True, on_step = False, on_epoch=True, logger=True)
        self.log("g_loss", gen_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("recon_loss", reconstruction_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("kld", kl_div, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("beta", beta, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("adv_loss", adversarial_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

    def on_train_epoch_start(self):
        self.adversarial_phase = True if self.current_epoch >= self.warmup else False
        return super().on_train_epoch_start()
    
    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        discr_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [gen_optimizer, discr_optimizer], []


class PQMFVAE(nn.Module):
    """
    PQMF-to-waveform VAE model based on IRCAM's RAVE:
    https://github.com/acids-ircam/RAVE/
    """
    def __init__(self,
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 48000,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [3, 3, 3, 3],
                 zdim: int = 128,
                 nchannels: int = 1,
                 strides: list[int] = [4, 4, 4, 2],
                 dilations: list[int] = [1, 3, 9],
                 nmog: int = 0):
        super().__init__()

        self.pqmf = pqmf
        self.strides = strides
        self.sr = sample_rate

        # encoder
        self.pqmf2hid = EncoderStack(pqmf.n_band*nchannels, nkernels, kernel_sizes, strides)
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
                                                nchannels*pqmf.n_band,
                                                kernel_size=7),
                                      nn.BatchNorm1d(nchannels*pqmf.n_band))
        self.hid2loud = nn.Sequential(nn.ZeroPad1d(get_padding(3)),
                                        nn.Conv1d(nkernels[0], 1, kernel_size=3))
        
        # prior
        if nmog > 0:
            self.prior = MOGPrior(zdim, nmog)
        else:
            self.prior = None
    
    def encode(self, x):
        x_mb = self.pqmf(x)
        h = self.pqmf2hid(x_mb)
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        h = self.z2hid(z)
        x_wav, x_loud = self.hid2wave(h), self.hid2loud(h)

        x_mb = x_wav * x_loud
        x_hat = self.pqmf.inverse(x_mb)
        return x_hat
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)

        z_reparam = mu + sigma*epsilon
        x_hat = torch.tanh(self.decode(z_reparam))

        return x_hat, mu, sigma

class MOGPrior(nn.Module):
    def __init__(self, zdim: int, n_components: int):
        super().__init__()

        self.zdim = zdim
        self.n_components = n_components

        if n_components > 0:
            self.weight_logits = nn.Parameter(torch.randn(n_components))
            self.means = nn.Parameter(torch.randn(n_components, zdim))
            self.variances = nn.Parameter(torch.randn(n_components, zdim))
        
        else:
            self.weight_logits = None
            self.means = None
            self.variances = None
    
    def kld_estimate(self, post_mean, post_var):
        weights = torch.softmax(self.weight_logits, 0)

        kld_components = torch.stack([kld_component(mean.reshape(1, -1, 1), var.reshape(1, -1, 1), post_mean, post_var) for (mean, var) in zip(self.means, self.variances)], 1)
        exp_sum = torch.sum(weights.reshape(1, -1, 1, 1) * torch.exp(-kld_components), 1)
        elbo = -torch.log(exp_sum)

        return elbo
    
    def forward(self, post_mean, post_var):
        if self.weight_logits is not None:
            kld = self.kld_estimate(post_mean, post_var)
        else:
            kld = kld_component(1, 0, post_mean, post_var)
        
        return kld

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
            LinterConvTranspose1D(
                input_dim,
                output_dim,
                kernel_size=stride*2,
                stride=stride, 
                padding=(stride//2)+1)
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

class GenerationCallback(Callback):
    def __init__(self, block_length: int, test_file: str, out_dir: str, test_freq: int = 5):
        self.test_file = test_file
        self.test_freq = test_freq
        self.out_dir = out_dir
        self.block_length = block_length

        self.output_test = self.test_file is not None and self.out_dir is not None

    def on_train_start(self, trainer, pl_module):
        if self.output_test:
            try:
                os.mkdir(self.out_dir)
            except FileExistsError:
                pass
        return super().on_train_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.trainer.current_epoch
        if epoch % self.test_freq == 0 and self.output_test:
            test_wave = [load_mono(self.test_file, pl_module.model.sr)]
            test_segs = reshape_data(test_wave, pl_module.block_length).to(pl_module.device)

            with torch.inference_mode():
                reconstructed_wave = torch.cat([pl_module(seg.unsqueeze(0))[0][:,:,:self.block_length] for seg in test_segs], dim=2).squeeze(0)

            torchaudio.save(f"{self.out_dir}/epoch_{epoch}.wav", reconstructed_wave.cpu(), pl_module.model.sr)
        return super().on_train_epoch_end(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        if self.output_test:
            test_wave = [load_mono(self.test_file, pl_module.model.sr)]
            test_segs = reshape_data(test_wave, pl_module.block_length).to(pl_module.device)

            with torch.inference_mode():
                reconstructed_wave = torch.cat([pl_module(seg.unsqueeze(0))[0][:,:,:self.block_length] for seg in test_segs], dim=2).squeeze(0)

            torchaudio.save(f"{self.out_dir}/epoch_{pl_module.trainer.current_epoch}.wav", reconstructed_wave.cpu(), pl_module.model.sr)
        return super().on_train_end(trainer, pl_module)

class LinterConvTranspose1D(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_size: int,
                 stride: int, 
                 padding: int):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(input_dim,
                                   output_dim,
                                   kernel_size,
                                   stride=1,
                                   padding=padding)
    
    def forward(self, x):
        T = x.shape[-1]
        upsampled_T = T * self.stride
        x_stretch = F.interpolate(x, size=upsampled_T, mode="nearest")
        y = self.conv(x_stretch)

        return y

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

def cos_annealed_beta(current_step, total_steps):
    return 0.5*(1- math.cos(current_step * math.pi / total_steps))

def lin_annealed_beta(current_step, total_steps):
    return min(1.0, current_step / total_steps)

def warm_cos_annealed_beta(current_step, total_steps):
    return 0.5*(1 - math.cos(current_step * 8 * math.pi / total_steps)) if (current_step*8//total_steps) % 2 == 0 else 1

def kld_component(prior_mean: torch.Tensor, prior_var: torch.Tensor, post_mean: torch.Tensor, post_var: torch.Tensor):
    return torch.pow(post_mean - prior_mean, 2) + torch.pow(post_var / prior_var, 2) - torch.log(torch.pow(post_var / prior_var, 2)) - 1

if __name__ == "__main__":
    x = torch.randn(4, 1, 49153)
    pqmf = PQMF()
    print(pqmf.inverse(pqmf(x)).shape)
    # vae = SingleVAE(input_dim = 4800)
    vae = PQMFVAE(pqmf)
    x_hat, mu, sigma = vae(x)
    print(x_hat.shape, mu.shape, sigma.shape)