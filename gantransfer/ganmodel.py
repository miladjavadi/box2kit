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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ZeroPad1d(get_padding(3, 2)),
            nn.Conv1d(1024, 256, kernel_size=3, stride=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(5, 2)),
            nn.Conv1d(256, 64, kernel_size=5, stride=2),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(5, 2)),
            nn.ConvTranspose1d(64, 256, kernel_size=5, stride=2, output_padding=0),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.ZeroPad1d(get_padding(3, 2)),
            nn.ConvTranspose1d(256, 1024, kernel_size=3, stride=2, output_padding=1)
        )

    def forward(self, input):
        return self.main(input)

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

class DACGAN(pl.LightningModule):
    def __init__(self, codec, device, block_length_in_samples, output_block_length_in_samples, block_length_in_frames, lambda_embedding=1, stft_loss = dac.nn.loss.MultiScaleSTFTLoss([2048, 1024, 512, 256, 128, 64])):
        super().__init__()
        self.tensor_device = device

        self.real_label = 1
        self.fake_label = 0

        self.lambda_embedding = lambda_embedding

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
        # self.toggle_optimizer(discr_optimizer)

        # Z_transformed = self.generator(Z_query).detach()
        
        # with torch.no_grad():
        #     transformed_decoded = self.codec.decode(Z_transformed)

        # target = target[:,:,:transformed_decoded.shape[2]] # trim tail of target that is lost when decoding
        
        # d_real = self.discriminator(Z_query, target)
        # d_fake = self.discriminator(Z_query, transformed_decoded)

        # real_labels = torch.full(d_real.shape, self.real_label, device=d_real.device, dtype=torch.float32)
        # fake_labels = torch.full(d_fake.shape, self.fake_label, device=d_fake.device, dtype=torch.float32)
        
        # real_adversarial_loss = adversarial_loss_fn(d_real, real_labels)
        # fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

        # discr_loss = real_adversarial_loss + fake_adversarial_loss
        # self.log("discr_loss", discr_loss, prog_bar=True)
        # discr_optimizer.zero_grad()
        # self.manual_backward(discr_loss)
        # discr_optimizer.step()
        # self.untoggle_optimizer(discr_optimizer)

        # # train generator
        # self.toggle_optimizer(gen_optimizer)

        ### NO MORE GAN STUFF

        Z_transformed = self.generator(Z_query)

        with torch.no_grad():
            transformed_decoded = self.codec.decode(Z_transformed)

        target = target[:,:,:transformed_decoded.shape[2]] # trim tail of target that is lost when decoding

        embedding_loss = embedding_loss_fn(Z_transformed, Z_target)

        ### MORE GAN STUFF

        # d_fake = self.discriminator(Z_query, transformed_decoded)
        # fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

        # gen_loss = 1/fake_adversarial_loss + self.lambda_embedding * embedding_loss

        # trying stuff

        ### NO MORE GAN STUFF

        ### STFT STUFF
        stft_loss = self.stft_loss(AudioSignal(transformed_decoded, self.sr), AudioSignal(target, self.sr))
        gen_loss = stft_loss + self.lambda_embedding * embedding_loss
        ### NO MORE STFT STUFF

        self.log("gen_loss", gen_loss, prog_bar=True)
        # STFT
        self.log("stft_loss", stft_loss, prog_bar=True)
        self.log("emb_mse", embedding_loss, prog_bar=True)
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