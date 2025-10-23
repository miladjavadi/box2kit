import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import pytorch_lightning as pl
from audiotools import AudioSignal
from pytorch_lightning.callbacks import Callback
from box2kit.utils.load_data import load_mono, reshape_data
from box2kit.utils.loss import MultiScaleSpectralDistance

import dac
import torchaudio

class Generator(nn.Module):
    """
    CNN for transforming 1024-dimensional latent sequences.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # nn.ZeroPad1d(get_padding(3, 2)),
            nn.Conv1d(1024, 256, kernel_size=3, padding="same"),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(5, 2)),
            nn.Conv1d(256, 64, kernel_size=5, padding="same"),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, kernel_size=7, padding="same"),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding="same"),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(5, 2)),
            nn.Conv1d(64, 256, kernel_size=5, padding="same"),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(3, 2)),
            nn.Conv1d(256, 1024, kernel_size=3, padding="same")
        )

    def forward(self, z):
        return self.main(z)
    

class DiscriminatorV2(nn.Module):
    """
    STFT CNN discriminator.

    Adapted from DCGAN's SpecGAN:
    https://github.com/chrisdonahue/wavegan/blob/master/specgan.py.

    Args:
        input_dims (list of int): Dimensions of STFT inputs ([n fft bins, n frames]).
        nkernels (list of int): Number of output kernels for each convolutional layer.
        kernel_sizes (list of int): Kernel lengths for each convolutional layer.
        strides (list of int): Stride length at each convolutional layer.
    """
    def __init__(self,
                 input_dims: list[int],
                 nfft: int = None,
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [5, 5, 5, 5],
                 strides: list[int] = [2, 2, 2, 2]):
        super().__init__()
        self.input_dims = input_dims
        self.nkernels = nkernels
        self.strides = strides
        
        if nfft is not None:
            self.nfft = nfft
        else:
            self.nfft = (input_dims[0]-1)*2 # assumed even-numbered fft window
        
        # self.nkernels = nkernels
        # self.kernel_sizes = kernel_sizes
        # self.strides = strides

        self.downconv = DownConvolver(input_dims, nkernels, kernel_sizes, strides)
        self.conv2score = nn.Linear(nkernels[-1]*(input_dims[1]//np.prod(strides)), 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # STFT -> Discriminator score in (0, 1)

        h = self.downconv(x)

        # Flatten convolution outputs
        h_flat = h.reshape(h.shape[0], self.nkernels[-1]*(self.input_dims[1]//np.prod(self.strides)))

        score = self.sigmoid(self.conv2score(h_flat)) # bce
        # score = self.conv2score(h_flat) # hinge
        return score


class DownConvolver(nn.Module):
    def __init__(self,
                 input_dims: list[int],
                 nkernels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int]):
        super().__init__()

        self.net = [nn.Sequential(nn.ZeroPad1d(get_padding(kernel_sizes[0], strides[0])),
                                  nn.Conv1d(input_dims[0],
                                            nkernels[0],
                                            kernel_sizes[0],
                                            strides[0]),
                                  nn.LeakyReLU())]
        
        for i in range(1, len(nkernels)):
            self.net.append(nn.Sequential(nn.ZeroPad1d(get_padding(kernel_sizes[i], strides[i])),
                                          nn.Conv1d(nkernels[i-1],
                                                    nkernels[i],
                                                    kernel_sizes[i],
                                                    strides[i]),
                                          nn.BatchNorm1d(nkernels[i]),
                                          nn.LeakyReLU()))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class DACGANV2(pl.LightningModule):
    """
    PyTorch Lightning module for adversarially training a CNN to map target-domain latent sequences to output-domain latent sequences.

    Training uses datasets of *paired* target and output audio signals.

    Args:
        input_segment_length (int): Length of pre-encoded audio segments.
        output_segment_length (int): Length of generated/reconstructed audio segments.
        nframes (int): (Unused) Length of encoded latent sequences.
        spectrum_dims (list of int): Dimensions of STFTs passed to discriminator ([n fft bins, n frames]).
        nfft (int): (Unused) Number of FFT bins in STFTs passed to discriminator.
        beta (float): Weight of latent space distance loss in generator loss function.
        phi (float): Weight of adversarial loss in generator loss function.
        codec (DAC): Codec used to encode latent space representations.
        stft_loss_fn (nn.Module): Loss function to use for multi-scale STFT loss.
        mel_loss_fn (nn.Module): Loss function to use for Mel spectrogram loss.
        warmup (int): Number of epochs before discriminator is initialized.
        lr (float): Learning rate for Adam optimizers.
    """
    def __init__(self,
                 input_segment_length: int,
                 output_segment_length: int,
                 nframes: int,
                 spectrum_dims: list[int],
                 nfft: int = None,
                 beta: float = 1,
                 phi: float = 1,
                 codec: dac.DAC = dac.DAC.load(dac.utils.download()).to("cuda") if torch.cuda.is_available() else dac.DAC.load(dac.utils.download()).to("cpu"),
                 stft_loss_fn = MultiScaleSpectralDistance([4096, 2048, 1024, 512, 256, 128, 64]),
                 mel_loss_fn = dac.nn.loss.MelSpectrogramLoss(window_lengths=[32, 64, 128, 256, 512, 1024, 2048], n_mels = [5, 10, 20, 40, 80, 160, 320], mel_fmin=[0], mel_fmax=[None], loss_fn=nn.MSELoss()),
                 mb_stft_loss_fn = dac.nn.loss.MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16]),
                 warmup: int = 250,
                 lr = 1e-4):
        super().__init__()

        self.generator, self.discriminator = self.initialize_models(spectrum_dims, nfft)
        self.codec = codec
        self.sr = codec.sample_rate
        self.lr = lr

        self.stft_loss_fn = stft_loss_fn
        self.mb_stft_loss_fn = mb_stft_loss_fn
        self.mel_loss_fn = mel_loss_fn
        self.embedding_loss_fn = nn.MSELoss()
        # self.adversarial_loss_fn = hinge_loss
        self.adversarial_loss_fn = nn.BCELoss()

        self.beta = beta
        self.phi = phi
        self.warmup = warmup

        self.input_block_length = input_segment_length
        self.output_block_length = output_segment_length
        self.nframes = nframes

        self.adversarial_phase = False

        self.automatic_optimization = False

        # the objective of the discriminator is to return 1 for real target recordings, and 0 for synthesized ones
        self.real_label = 1
        self.fake_label = 0

        self.codec.eval()
        self.codec.requires_grad_(False)

        self.save_hyperparameters(ignore=["codec", "spectral_loss_fn", "mel_loss_fn", "stft_loss_fn", "mb_stft_loss_fn"])
    
    def initialize_models(self, spectrum_dims: list[int], nfft: int = None):
        generator = Generator()
        discriminator = DiscriminatorV2(spectrum_dims, nfft=nfft)
        return generator, discriminator
    
    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        _, output_audio, target_latents, output_latents = batch

        gen_optimizer, discr_optimizer = self.optimizers()

        # train generator
        self.toggle_optimizer(gen_optimizer)

        gen_latents = self.generator(target_latents)

        with torch.no_grad():
            gen_audio = self.codec.decode(gen_latents)
            output_trim = output_audio[:,:,:gen_audio.shape[-1]] # trim to match dims
            stft_gen = torch.stft(gen_audio.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=gen_audio.device), return_complex=True).abs()
            stft_output = torch.stft(output_trim.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=output_trim.device), return_complex=True).abs()
        
        # calculate generator losses
        gen_optimizer.zero_grad()

        spectral_loss = self.stft_loss_fn(gen_audio, output_trim)# + self.mb_stft_loss_fn(gen_mb_AS, output_mb_AS)# + self.mel_loss_fn(gen_AS, output_AS)
        # spectral_loss = self.mel_loss_fn(gen_AS, output_AS)
        embedding_loss = self.embedding_loss_fn(gen_latents, output_latents)

        if self.adversarial_phase:
            gen_score = self.discriminator(stft_gen)
            real_labels = torch.full_like(gen_score, fill_value=self.real_label)
            # # how convinced the discriminator is that generated waveforms are real
            adversarial_loss = self.adversarial_loss_fn(gen_score, real_labels) # bce loss
            # adversarial_loss = torch.mean(gen_score) # hinge loss

        else:
            adversarial_loss = 0

        generator_loss = spectral_loss + self.beta * embedding_loss + self.phi * adversarial_loss
        self.manual_backward(generator_loss)
        gen_optimizer.step()
        self.untoggle_optimizer(gen_optimizer)

        # train discriminator
        if self.adversarial_phase:
            self.toggle_optimizer(discr_optimizer)
            discr_optimizer.zero_grad()
            # how convinced the discriminator is that target waveforms are real, and generated waveforms are fake
            real_score = self.discriminator(stft_output)
            gen_score = self.discriminator(stft_gen.detach())

            real_labels = torch.full_like(real_score, fill_value=self.real_label)
            fake_labels = torch.full_like(gen_score, fill_value=self.fake_label)

            discr_loss = self.adversarial_loss_fn(real_score, real_labels) + self.adversarial_loss_fn(gen_score, fake_labels)

            # discr_loss = self.adversarial_loss_fn(real_score, self.real_label) + self.adversarial_loss_fn(gen_score, self.fake_label)
            self.manual_backward(discr_loss)
            discr_optimizer.step()
            self.untoggle_optimizer(discr_optimizer)
        else:
            discr_loss = 0
        
        self.log("d_loss", discr_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("g_loss", generator_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("spectral_loss", spectral_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("embedding_loss", embedding_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("adversarial_loss", adversarial_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
    
    def validation_step(self, batch):
        _, output_audio, target_latents, output_latents = batch
        
        gen_latents = self.generator(target_latents)

        with torch.no_grad():
            gen_audio = self.codec.decode(gen_latents)
            output_trim = output_audio[:,:,:gen_audio.shape[-1]]

        val_loss = self.stft_loss_fn(gen_audio, output_trim)# + self.mb_stft_loss_fn(gen_mb_AS, output_mb_AS)# + self.mel_loss_fn(gen_AS, output_AS)

        self.log("val_loss", val_loss, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        discr_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [gen_optimizer, discr_optimizer], []
    
    def on_train_epoch_start(self):
        self.codec.eval()
        self.adversarial_phase = True if self.current_epoch >= self.warmup else False
        return super().on_train_epoch_start()

    def on_fit_epoch_start(self):
        self.codec.eval()
    
    def on_validation_epoch_start(self):
        self.codec.eval()


class GenerationCallback(Callback):
    """
    Periodically generate output audio data from a test target audio file.

    Args:
        segment_length (int): Segment length to divide audio into.
        test_file (str): Path to test target audio file.
        out_dir (str): Path to directory for saving generated output audio.
        test_freq (int): How often, in epochs, to generate output audio. 
    """
    def __init__(self, test_file: str, out_dir: str, test_freq: int = 5):
        self.test_file = test_file
        self.test_freq = test_freq
        self.out_dir = out_dir

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
        codec = pl_module.codec

        if epoch % self.test_freq == 0 and self.output_test:
            test_wave = [load_mono(self.test_file, codec.sample_rate)]
            test_segs = reshape_data(test_wave, pl_module.input_block_length).to(codec.device)

            with torch.inference_mode():
                test_latents = codec.encode(test_segs)[0]
                test_out_latents = pl_module(test_latents)
                test_out_segs = codec.decode(test_out_latents)[:,:pl_module.output_block_length]
                reconstructed_wave = test_out_segs.reshape(1, -1)

            torchaudio.save(f"{self.out_path}/epoch_{epoch}.wav", reconstructed_wave.cpu(), codec.sample_rate)
        return super().on_train_epoch_end(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        epoch = pl_module.trainer.current_epoch
        codec = pl_module.codec

        if self.output_test:
            test_wave = [load_mono(self.test_file, codec.sample_rate)]
            test_segs = reshape_data(test_wave, pl_module.input_block_length).to(codec.device)

            with torch.inference_mode():
                test_latents = codec.encode(test_segs)[0]
                test_out_latents = pl_module(test_latents)
                test_out_segs = codec.decode(test_out_latents)[:,:pl_module.output_block_length]
                reconstructed_wave = test_out_segs.reshape(1, -1)

            torchaudio.save(f"{self.out_path}/epoch_{epoch}.wav", reconstructed_wave.cpu(), codec.sample_rate)
        return super().on_train_epoch_end(trainer, pl_module)
        

def get_padding(kernel_size, stride=1, dilation=1):
    effective_kernel = (kernel_size - 1) * dilation + 1
    pad_total = max(effective_kernel - stride, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return (pad_left, pad_right)


def hinge_loss(score: torch.Tensor, label: float):
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
    x = torch.randn(4, 1024, 10000)
    discr = DiscriminatorV2([x.shape[1], x.shape[2]])

    score = discr(x)
    print(x.shape)