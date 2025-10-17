import torch
import numpy as np
import math
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torchaudio
from box2kit.utils.load_data import load_dir, reshape_data, load_mono
from box2kit.waveenc.rave_pqmf import PQMF
from audiotools import AudioSignal
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from box2kit.neural.model import DiscriminatorV2
import os

class LightningVAE(pl.LightningModule):
    def __init__(self,
                 block_length: int,
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 44100,
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
    

class TransferGAN(LightningVAE):
    def __init__(self,
                 block_length: int,
                 discriminator_dims: list[int],
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 44100,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [3, 3, 3, 3],
                 zdim: int = 128,
                 nchannels: int = 1,
                 strides: list[int] = [4, 4, 4, 2],
                 dilations: list[int] = [1, 3, 9],
                 nmog: int = 0,
                 mel_loss_fn = MelSpectrogramLoss(window_lengths=[32, 64, 128, 256, 512, 1024, 2048], n_mels = [5, 10, 20, 40, 80, 160, 320], mel_fmin=[0,0,0,0,0,0,0], mel_fmax=[None,None,None,None,None,None,None]),
                 full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[2048, 1024, 512, 256, 128]),
                 mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16]),
                 lr: float = 1e-4,
                 phi: float = 1,
                 warmup: int = 250,
                 beta: float = 1
                 ):
        self.lambda_adversarial = phi
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
        self.adversarial_loss_fn = nn.BCELoss()
        # self.adversarial_loss_fn = hinge_loss
        self.warmup = warmup
        self.beta_max = beta

        self.automatic_optimization = False
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        gen_optimizer, discr_optimizer = self.optimizers()

        # train generator
        self.toggle_optimizer(gen_optimizer)

        y_hat, mu, log_var = self.model(x)
        
        y_hat = y_hat[:,:,:y.shape[2]] # the model works on frames of length (strides x pqmf_bands) = 4x4x4x2x16 = 2048 samples

        # generator losses
        y_hat_AS, y_AS = AudioSignal(y_hat, self.model.sr), AudioSignal(y, self.model.sr)
        fullband_reconstruction_loss = self.mel_loss_fn(y_hat_AS, y_AS) # + self.full_stft_loss_fn(y_hat_AS, y_AS)

        multiband_reconstruction_loss = self.mb_stft_loss_fn(AudioSignal(self.model.pqmf(critical_pad(y, 16)), self.model.sr), AudioSignal(self.model.pqmf(critical_pad(y, 16)), self.model.sr))
        reconstruction_loss = fullband_reconstruction_loss + multiband_reconstruction_loss

        kl_div = torch.mean(self.model.prior(mu, log_var), dim=0) 

        if self.adversarial_phase:
            stft_gen = torch.stft(y_hat.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=y_hat.device), return_complex=True).abs()
            gen_score = self.discriminator(stft_gen)
            real_labels = torch.full_like(gen_score, fill_value=self.real_label) # bce
            # # how convinced the discriminator is that generated waveforms are real
            adversarial_loss = self.adversarial_loss_fn(gen_score, real_labels) # bce
            # adversarial_loss = torch.mean(gen_score) # hinge

        else:
            adversarial_loss = 0

        # backprop
        beta = self.beta_max * cos_annealed_beta(self.trainer.current_epoch, self.trainer.max_epochs/4)
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

            # discr_loss = self.adversarial_loss_fn(real_score, self.real_label) + self.adversarial_loss_fn(gen_score, self.fake_label) # hinge
            discr_loss = self.adversarial_loss_fn(real_score, real_labels) + self.adversarial_loss_fn(gen_score, fake_labels) # bce
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
        if self.adversarial_phase:
            # freeze encoder during adversarial phase
            self.model.pqmf2hid.requires_grad_ = False
            self.model.hid2mu.requires_grad_ = False
            self.model.hid2sigma.requires_grad_ = False
            self.model.prior.requires_grad_ = False
        return super().on_train_epoch_start()
    
    def validation_step(self, batch):
        x, y = batch

        y_hat, mu, log_var = self.model(x)
        y_hat = y_hat[:,:,:y.shape[2]] # the model works on frames of length (strides x pqmf_bands) = 4x4x4x2x16 = 2048 samples

        # generator losses
        y_hat_AS, y_AS = AudioSignal(y_hat, self.model.sr), AudioSignal(y, self.model.sr)
        val_loss = self.mel_loss_fn(y_hat_AS, y_AS)

        self.log("val_loss", val_loss, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        discr_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [gen_optimizer, discr_optimizer], []


# current
class PQMFVAE(nn.Module):
    """
    PQMF-to-waveform VAE model based on IRCAM's RAVE:
    https://github.com/acids-ircam/RAVE/
    """
    def __init__(self,
                 pqmf: PQMF = PQMF(),
                 sample_rate: int = 44100,
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

        self.hid2noise = NoiseGenerator(in_size=nkernels[0],
                                        hidden_size=128,
                                        data_size=pqmf.n_band,
                                        strides=[2, 2, 2],
                                        noise_bands=5)
        
        # prior
        # if nmog > 0:
        #     self.prior = MOGPrior(zdim, nmog)
        # else:
        #     self.prior = None
        self.prior = MOGPrior(zdim, nmog)

    def encode(self, x):
        x_pad = critical_pad(x, 16)
        # x_pad = x[..., :((x.shape[-1]//16)*16)]
        x_mb = self.pqmf(x_pad)

        h = self.pqmf2hid(x_mb)

        mu, log_var = self.hid2mu(h), self.hid2sigma(h)
        return mu, log_var
    
    def decode(self, z):
        h = self.z2hid(z)
        x_wav, x_loud, x_noise = self.hid2wave(h), self.hid2loud(h), self.hid2noise(h)

        x_mb = x_wav * x_loud + x_noise
        x_hat = self.pqmf.inverse(x_mb)
        return x_hat
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        epsilon = torch.randn_like(log_var)

        z_reparam = mu + torch.sqrt(torch.exp(log_var))*epsilon
        x_hat = torch.tanh(self.decode(z_reparam))

        return x_hat, mu, log_var


class MOGPrior(nn.Module):
    def __init__(self, zdim: int, n_components: int):
        super().__init__()

        self.zdim = zdim
        self.n_components = n_components

        if n_components > 0:
            self.weight_logits = nn.Parameter(torch.ones(n_components))
            self.means = nn.Parameter(torch.zeros(n_components, zdim))
            self.log_vars = nn.Parameter(torch.zeros(n_components, zdim))
        
        else:
            self.weight_logits = None
            self.means = None
            self.log_vars = None
    
    def kld_estimate(self, post_mean, post_log_var):
        log_weights = torch.log_softmax(self.weight_logits, 0)
        kld_components = torch.stack([kld_component(mean.reshape(1, -1, 1), log_var.reshape(1, -1, 1), post_mean, post_log_var) for (mean, log_var) in zip(self.means, self.log_vars)]) # [M, B, T]
        elbo = -torch.logsumexp(log_weights.reshape(-1, 1, 1) - kld_components, dim=0)
        return elbo
    
    def forward(self, post_mean, post_log_var):
        if self.weight_logits is not None:
            kld = torch.mean(self.kld_estimate(post_mean, post_log_var), dim=-1)
        else:
            # if no trainable prior is used, use closed-form kld expression with standard gaussian prior
            kld = torch.mean(kld_component(0, 0, post_mean, post_log_var), dim=-1)
        
        return kld # [B]


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


class NoiseGenerator(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        data_size: int,
        strides: list[int],
        noise_bands: int,
        n_channels: int = 1,
        activation = nn.LeakyReLU(.2)
    ):
        super().__init__()
        net = []
        self.n_channels = n_channels
        self.target_size = np.prod(strides)
        channels = [in_size]
        channels.extend((len(strides) - 1) * [hidden_size])
        channels.append(data_size * noise_bands * n_channels)

        for i, r in enumerate(strides):
            net.extend([
                nn.ZeroPad1d([r, 0]),
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    2 * r,
                    stride=r,
                )])
            if i != len(strides) - 1:
                net.append(activation)

        self.net = nn.Sequential(*net)
        self.data_size = data_size

    def forward(self, x):
        in_len = x.shape[-1]
        # noise synthesis will be truncated to nearest multiple of self.target_size.
        # to compensate, we first pad signal to next multiple...
        x = critical_pad(x, self.target_size)
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.n_channels * self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)

        # ...and at the end, re-trim it to its original length
        noise = noise[..., :in_len]
        return noise


class GenerationCallback(Callback):
    def __init__(self, block_length: int, test_file: str, out_dir: str, test_freq: int = 5):
        self.test_file = test_file
        self.test_freq = test_freq
        self.out_dir = out_dir
        self.block_length = block_length

        self.output_test = self.test_file is not None and self.out_dir is not None

    def on_train_start(self, trainer, pl_module):
        log_path = trainer.logger.log_dir
        if self.output_test:
            try:
                os.mkdir(os.path.join(log_path, self.out_dir))
            except FileExistsError:
                pass
            self.out_path = os.path.join(log_path, self.out_dir)
        return super().on_train_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.trainer.current_epoch
        if epoch % self.test_freq == 0 and self.output_test:
            test_wave = [load_mono(self.test_file, pl_module.model.sr)]
            test_segs = reshape_data(test_wave, pl_module.block_length).to(pl_module.device)

            with torch.inference_mode():
                reconstructed_wave = torch.cat([pl_module(seg.unsqueeze(0))[0][:,:,:self.block_length] for seg in test_segs], dim=2).squeeze(0)

            torchaudio.save(f"{self.out_path}/epoch_{epoch}.wav", reconstructed_wave.cpu(), pl_module.model.sr)
        return super().on_train_epoch_end(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        if self.output_test:
            test_wave = [load_mono(self.test_file, pl_module.model.sr)]
            test_segs = reshape_data(test_wave, pl_module.block_length).to(pl_module.device)

            with torch.inference_mode():
                reconstructed_wave = torch.cat([pl_module(seg.unsqueeze(0))[0][:,:,:self.block_length] for seg in test_segs], dim=2).squeeze(0)

            torchaudio.save(f"{self.out_path}/epoch_{pl_module.trainer.current_epoch}.wav", reconstructed_wave.cpu(), pl_module.model.sr)
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


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, mode = "centered") -> tuple[int]:
        """
        Compute non-symmetric padding amount for preserving convolution output shape.

        Copied from cached_conv by IRCAM:
        https://github.com/acids-ircam/cached_conv/blob/master/cached_conv/convs.py

        Args:
            kernel_size (int): Size of convolution kernel used.
            stride (int): Stride length used in convolution.
            dilation (int): Dilation amount used in convolution.
            mode (str): Shift padding to achieve different modes of causality. One of:
                "centered": Pseudo-equal padding on both sides.
                "causal": Padding at beginning only.
                "anticausal": Padding at end only.
            
        Returns:
            p_left (int): Padding amount at beginning of signal.
            p_right (int): Padding amount at end of signal.
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


def cos_annealed_beta(current_step: int, n_warmup_steps: int) -> float:
    return 0.5*(1- math.cos(current_step * math.pi / n_warmup_steps)) if current_step < n_warmup_steps else 1


def lin_annealed_beta(current_step: int, total_steps: int) -> float:
    return min(1.0, current_step / total_steps)


def warm_cos_annealed_beta(current_step: int, total_steps: int) -> float:
    return 0.5*(1 - math.cos(current_step * 8 * math.pi / total_steps)) if (current_step*8//total_steps) % 2 == 0 else 1


def kld_component(prior_mean: torch.Tensor, prior_log_var: torch.Tensor, post_mean: torch.Tensor, post_log_var: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.pow(post_mean - prior_mean, 2) + torch.exp(post_log_var - prior_log_var) - (post_log_var - prior_log_var) - 1, (-2))


def mod_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def amp_to_impulse_response(amp: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Transform filterbank weights to impulse responses on dim -1.
    
    Copied from IRCAM ACID'S RAVE:
    https://github.com/acids-ircam/RAVE/blob/master/rave/core.py

    Args:
        amp (Tensor): Filterbank weights.
        target_size (int): Desired impulse response length.
    
    Output:
        amp (Tensor): Impulse response.
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve signal with kernel on dim -1.

    Copied from IRCAM ACIDS' RAVE:
    https://github.com/acids-ircam/RAVE/blob/master/rave/core.py

    Args:
        signal (Tensor): Signal to convolve.
        kernel (Tensor): Kernel to convolve with.
    
    Returns:
        output (Tensor): Convolution output.
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def hinge_loss(score: torch.Tensor, label: float) -> torch.Tensor:
    """
    Return hinge loss from discriminator score-label discrepancy.

    Args:
        score (Tensor): Discriminator scores.
        label (float): Labeled scores.
    
    Returns:
        Hinge loss.
    """

    zeros = torch.zeros_like(score)
    if label > 0:
        return torch.mean(torch.min(zeros, label - score))
    else:
        return torch.mean(torch.min(zeros, -label + score))
    

def critical_pad(signal: torch.Tensor, divisor: int) -> torch.Tensor:
    """
    Zero-pad dim -1 of a tensor such that its total length is perfectly divisible by an integer divisor.
    
    Useful for making signals of arbitrary lengths compatible with critically sampled filterbanks, such as PQMF.

    Args:
        signal (Tensor): Tensor to pad.
        divisor (int): Divisor with which to make signal compatible.
    
    Returns:
        Padded signal.
    """

    return torch.nn.functional.pad(signal, [0, divisor-(signal.shape[-1] % divisor)])


if __name__ == "__main__":
    x = critical_pad(torch.randn(4, 1, 48004), 16)
    pqmf = PQMF()
    print(pqmf.inverse(pqmf(x)).shape)
    # vae = SingleVAE(input_dim = 4800)
    vae = PQMFVAE(pqmf)
    x_hat, mu, log_var = vae(x)
    # print(x_hat.shape, mu.shape, log_var.shape)
    print(torch.isnan(mu).any())