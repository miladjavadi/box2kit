import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np

import dac
import torchaudio

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1024, 256, kernel_size=5, padding="same", padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv1d(256, 64, kernel_size=5, padding="same", padding_mode="reflect"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 256, kernel_size=5, padding="same", padding_mode="reflect"),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 1024, kernel_size=5, padding="same", padding_mode="reflect")
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
            nn.Conv1d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.waveform_path = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        self.combined_layer = nn.Sequential(
            nn.Linear((64*(block_length_in_frames//4) + 64*(block_length_in_samples//8)), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
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
    
    def __getitem__(self, idx):
        x = self.query_data[idx]
        y = self.target_data[idx]
        return x, y