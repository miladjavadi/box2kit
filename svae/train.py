import torch
from tqdm import tqdm
from torch import nn, optim
from svae.model import SingleVAE, WaveSegmentDataset
import torchaudio
from torch.utils.data import DataLoader
# from utils.load_data import load_dir, load_mono
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from audiotools import AudioSignal

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_DIM = 200
Z_DIM = 8
NUM_EPOCHS = 4
BATCH_SIZE = 8
LR = 1e-4
SAMPLE_RATE = 48000
TEMPO = 90
SUBDIVS = 8

block_length = int(SAMPLE_RATE*60/(TEMPO*SUBDIVS/4))

dataset_path = "training_data/drum_kit"

dataset = WaveSegmentDataset(dataset_path, block_length, SAMPLE_RATE)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = SingleVAE(block_length, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
reconstruction_loss_fn = MelSpectrogramLoss()

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, x in loop:
        # forward
        x = x.to(DEVICE)
        x_hat, mu, sigma = model(x)

        # losses
        reconstruction_loss = reconstruction_loss_fn(AudioSignal(x_hat, SAMPLE_RATE), AudioSignal(x, SAMPLE_RATE))
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix({"loss": loss.item(),
                          "recon_loss": reconstruction_loss.item(), 
                          "kl_div": kl_div.item()})
        

