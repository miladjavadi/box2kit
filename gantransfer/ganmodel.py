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

import dac
import torchaudio

### GENERATOR

class Generator(nn.Module):
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
            # nn.ZeroPad1d(get_padding(5, 2)),
            nn.Conv1d(64, 256, kernel_size=5, padding="same"),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(3, 2)),
            nn.Conv1d(256, 1024, kernel_size=3, padding="same")
        )

    def forward(self, input):
        return self.main(input)
    
### DISCRIMINATOR

class Discriminator(nn.Module):
    # generate discriminator score between 0 and 1.
    def __init__(self, block_length_in_samples, block_length_in_frames):
        super(Discriminator, self).__init__()

        self.block_length_in_samples = block_length_in_samples
        self.block_length_in_frames = block_length_in_frames

        self.embedding_path = nn.Sequential(
            nn.Conv1d(1024, 2, kernel_size=4, stride=4, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(2, 1, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
        )

        self.waveform_path = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=4, stride=4, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(2, 2, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(2, 1, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )

        self.combined_layer = nn.Sequential(
            nn.Linear((1*(block_length_in_frames//16) + 1*(block_length_in_samples//64)), 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embedding, waveform):
        embedding = embedding.to(torch.float32)
        waveform = waveform.to(torch.float32)

        embedding_features = self.embedding_path(embedding)
        waveform_features = self.waveform_path(waveform)

        combined_features = torch.cat([embedding_features.flatten(1, -1), waveform_features.flatten(1, -1)], dim=1)
        output_probability = self.combined_layer(combined_features)


        return output_probability

class DiscriminatorV2(nn.Module):
    """
    Adapted from DCGAN's SpecGAN:
    https://github.com/chrisdonahue/wavegan/blob/master/specgan.py
    """
    def __init__(self,
                 input_dims: list[int],
                 nkernels: list[int] = [64, 128, 256, 512],
                 kernel_sizes: list[int] = [5, 5, 5, 5],
                 strides: list[int] = [2, 2, 2, 2]):
        super().__init__()
        self.input_dims = input_dims # STFT dims ([nfft, nframes])
        self.nkernels = nkernels
        self.strides = strides
        self.nfft = input_dims[0]
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

        score = self.sigmoid(self.conv2score(h_flat))
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

### DATASET

class PairedWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, target_data):
        if query_data.shape != target_data.shape:
            raise Exception(f"Query dataset and target dataset must have the same size (query dataset has shape {query_data.shape}, while target dataset has shape {target_data.shape})")
        self.query_data = query_data
        self.target_data = target_data
    
    def __len__(self):
        return self.query_data.shape[0]
    
    def __getitem__(self, idx: int):
        x = self.query_data[idx]
        y = self.target_data[idx]
        return x, y

### LIGHTNING MODULE

class DACGAN(pl.LightningModule):
    def __init__(self, codec, device, block_length_in_samples, output_block_length_in_samples, block_length_in_frames, lambda_embedding=1, lambda_stft=1, stft_loss = dac.nn.loss.MultiScaleSTFTLoss([2048, 1024, 512, 256, 128, 64])):
        super().__init__()
        self.tensor_device = device

        self.real_label = 1
        self.fake_label = 0

        self.lambda_embedding = lambda_embedding
        self.lambda_stft = lambda_stft

        self.block_length_in_samples = block_length_in_samples # needed for inference

        self.stft_loss = stft_loss
        
        # needed to re-initialize discriminator
        self.output_block_length_in_samples = output_block_length_in_samples
        self.block_length_in_frames = block_length_in_frames

        self.generator, self.discriminator = self.initialize_models()
        self.codec = codec
        # self.real_labels = torch.full((1), real_label, device=device, dtype=torch.float32)
        # self.fake_labels = torch.full((1), fake_label, device=device, dtype=torch.float32)
        self.sr = codec.sample_rate

        self.automatic_optimization = False

        self.codec.eval()
        self.codec.requires_grad_(False)

        self.save_hyperparameters("block_length_in_samples", "lambda_embedding", "output_block_length_in_samples", "block_length_in_frames")

    def initialize_models(self):
        gen_model = Generator().to(self.tensor_device)
        discr_model = Discriminator(self.output_block_length_in_samples, self.block_length_in_frames).to(self.tensor_device)
        # codec = dac.DAC.load(dac.utils.download()).to(self.tensor_device)

        return gen_model, discr_model#, codec

    def forward(self, input):
        return self.generator(input)
    
    def training_step(self, batch):
        input_waveforms = batch
        query, target = input_waveforms

        gen_optimizer, discr_optimizer = self.optimizers()

        embedding_loss_fn = nn.MSELoss()
        adversarial_loss_fn = nn.BCELoss()

        with torch.no_grad():
            Z_query = self.codec.encode(query)[0]
            Z_target = self.codec.encode(target)[0]

        ### GAN STUFF

        # train discriminator
        self.toggle_optimizer(discr_optimizer)

        Z_transformed = self.generator(Z_query).detach()
        
        with torch.no_grad():
            transformed_decoded = self.codec.decode(Z_transformed)

        target = target[:,:,:transformed_decoded.shape[2]] # trim tail of target that is lost when decoding
        
        d_real = self.discriminator(Z_query, target)
        d_fake = self.discriminator(Z_query, transformed_decoded)

        real_labels = torch.full(d_real.shape, self.real_label, device=d_real.device, dtype=torch.float32)
        fake_labels = torch.full(d_fake.shape, self.fake_label, device=d_fake.device, dtype=torch.float32)
        
        real_adversarial_loss = adversarial_loss_fn(d_real, real_labels)
        fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

        discr_loss = real_adversarial_loss + fake_adversarial_loss
        self.log("discr_loss", discr_loss, prog_bar=False)
        discr_optimizer.zero_grad()
        self.manual_backward(discr_loss)
        discr_optimizer.step()
        self.untoggle_optimizer(discr_optimizer)

        # train generator
        self.toggle_optimizer(gen_optimizer)

        ### NO MORE GAN STUFF

        Z_transformed = self.generator(Z_query)

        with torch.no_grad():
            transformed_decoded = self.codec.decode(Z_transformed)

        # target = target[:,:,:transformed_decoded.shape[2]] # trim tail of target that is lost when decoding

        embedding_loss = embedding_loss_fn(Z_transformed, Z_target)

        ### MORE GAN STUFF

        d_fake = self.discriminator(Z_query, transformed_decoded)
        fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

        # gen_loss = 1/fake_adversarial_loss + self.lambda_embedding * embedding_loss

        ### NO MORE GAN STUFF

        ### STFT STUFF
        stft_loss = self.stft_loss(AudioSignal(transformed_decoded, self.sr), AudioSignal(target, self.sr))
        gen_loss = self.lambda_stft * stft_loss + self.lambda_embedding * embedding_loss + 1/fake_adversarial_loss
        ### NO MORE STFT STUFF

        self.log("gen_loss", gen_loss, prog_bar=True)
        # STFT
        self.log("stft_loss", stft_loss, prog_bar=True)
        self.log("emb_mse", embedding_loss, prog_bar=True)
        self.log("fake_discr_loss", fake_adversarial_loss, prog_bar=True)
        gen_optimizer.zero_grad()
        self.manual_backward(gen_loss)
        gen_optimizer.step()
        self.untoggle_optimizer(gen_optimizer)
    
    def validation_step(self, batch):
        pass
    
    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
        discr_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
        return [gen_optimizer, discr_optimizer], []
    
    def on_train_epoch_start(self):
        self.codec.eval()

    def on_fit_epoch_start(self):
        self.codec.eval()
    
    def on_validation_epoch_start(self):
        self.codec.eval()
    
    # def on_save_checkpoint(self, checkpoint):
    #     # Remove the evaluator's weights from the checkpoint's state_dict
    #     keys_to_remove = [k for k in checkpoint['state_dict'] if k.startswith('codec.')]
    #     for k in keys_to_remove:
    #         del checkpoint['state_dict'][k]

class DACGANV2(pl.LightningModule):
    def __init__(self,
                 input_block_length: int,
                 output_block_lengths: int,
                 nframes: int,
                 spectrum_dims: list[int],
                 lambda_embedding: float = 1,
                 lambda_adversarial: float = 1,
                 codec: dac.DAC = dac.DAC.load(dac.utils.download()).to("cuda") if torch.cuda.is_available() else dac.DAC.load(dac.utils.download()).to("cpu"),
                 spectral_loss_fn = dac.nn.loss.MultiScaleSTFTLoss([2048, 1024, 512, 256, 128, 64]),
                 warmup: int = 250):
        super().__init__()

        self.generator, self.discriminator = self.initialize_models(spectrum_dims)
        self.codec = codec
        self.sr = codec.sample_rate

        self.spectral_loss_fn = spectral_loss_fn
        self.embedding_loss_fn = nn.MSELoss()
        self.adversarial_loss_fn = nn.BCELoss()

        self.lambda_embedding = lambda_embedding
        self.lambda_adversarial = lambda_adversarial
        self.warmup = warmup

        self.adversarial_phase = False
        # the objective of the discriminator is to return 1 for real target recordings, and 0 for synthesized ones
        self.real_label = torch.ones(1, dtype=torch.float32)
        self.fake_label = torch.zeros(1, dtype=torch.float32)

        self.automatic_optimization = False
    
    def initialize_models(self, spectrum_dims: list[int]):
        generator = Generator()
        discriminator = DiscriminatorV2(spectrum_dims)
        return generator, discriminator
    
    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx):
        input_waveforms = batch
        query, target = input_waveforms

        gen_optimizer, discr_optimizer = self.optimizers()

        with torch.no_grad():
            Z_query = self.codec.encode(query)[0]
            Z_target = self.codec.encode(target)[0]

        # train generator
        self.toggle_optimizer(gen_optimizer)

        Z_gen = self.generator(Z_query)

        with torch.no_grad():
            gen = self.codec.decode(Z_gen)
            target_post = self.codec.decode(Z_target) # use post-dac target to match dims
            stft_gen = torch.stft(gen.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=gen.device)).abs()
            stft_target = torch.stft(target_post.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=target_post.device)).abs()
        
        # calculate generator losses
        gen_optimizer.zero_grad()

        spectral_loss = self.spectral_loss_fn(AudioSignal(gen, self.sr), AudioSignal(target_post, self.sr))
        embedding_loss = self.embedding_loss_fn(Z_gen, Z_target)

        if self.adversarial_phase:
            # how convinced the discriminator is that generated waveforms are real
            adversarial_loss = self.adversarial_loss_fn(self.discriminator(stft_gen), self.real_label)

        else:
            adversarial_loss = 0

        generator_loss = spectral_loss + self.lambda_embedding * embedding_loss + self.lambda_adversarial * adversarial_loss
        self.manual_backward(generator_loss)
        gen_optimizer.step()
        self.untoggle_optimizer(gen_optimizer)

        # train discriminator
        if self.adversarial_phase:
            self.toggle_optimizer(discr_optimizer)
            discr_optimizer.zero_grad()
            # how convinced the discriminator is that target waveforms are real, and generated waveforms are fake
            discr_loss = self.adversarial_loss_fn(self.discriminator(stft_target), self.real_label) + self.adversarial_loss_fn(self.discriminator(stft_gen.detach()), self.fake_label)
            self.manual_backward(discr_loss)
            discr_optimizer.step()
            self.untoggle_optimizer(discr_optimizer)
    
    def validation_step(self, batch):
        pass
    
    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
        discr_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))
        return [gen_optimizer, discr_optimizer], []
    
    def on_train_epoch_start(self):
        self.codec.eval()
        self.adversarial_phase = True if self.current_epoch >= self.warmup else False
        return super().on_train_epoch_start()

    def on_fit_epoch_start(self):
        self.codec.eval()
        super().on_fit_epoch_start()
    
    def on_validation_epoch_start(self):
        self.codec.eval()
        super().on_fit_epoch_start()
        

### UTILITY FUNCTIONS

def get_padding(kernel_size, stride=1, dilation=1):
    effective_kernel = (kernel_size - 1) * dilation + 1
    pad_total = max(effective_kernel - stride, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return (pad_left, pad_right)

if __name__ == "__main__":
    x = torch.randn(4, 1024, 10000)
    discr = DiscriminatorV2([x.shape[1], x.shape[2]])

    score = discr(x)
    print(x.shape)