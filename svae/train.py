import torch
from tqdm import tqdm
from torch import nn, optim
from svae.model import SingleVAE, WaveSegmentDataset, PQMFVAE
import torchaudio
from torch.utils.data import DataLoader
from utils.load_data import load_dir, load_mono, reshape_data
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from audiotools import AudioSignal
from torch.utils.tensorboard import SummaryWriter
import argparse
import math
import os
from svae.pqmf import PQMF

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 48000

def lin_annealed_beta(current_step, total_steps):
    return min(1.0, current_step / total_steps)

def cos_annealed_beta(current_step, total_steps):
    return 0.5*(1 - math.cos(current_step * 8 * math.pi / total_steps)) if (current_step*8//total_steps) % 2 == 0 else 1

def main(args):
    H_DIM = args.hdim
    Z_DIM = args.zdim
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    LR = args.lr
    TEMPO = args.bpm
    SUBDIVS = args.subdivs
    TEST_FILE = args.test
    TEST_OUT = args.out
    TEST_FREQ = args.outfreq

    block_length = int(SAMPLE_RATE*60/(TEMPO*SUBDIVS/4))
    # trunc_block_length = (block_length//2048)*2048
    # remainder = block_length % 2048


    dataset_path = args.data

    dataset = WaveSegmentDataset(dataset_path, block_length, SAMPLE_RATE)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    pqmf = PQMF()
    # model = SingleVAE(block_length, H_DIM, Z_DIM).to(DEVICE)
    model = PQMFVAE(pqmf).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=0.0, last_epoch=-1)

    # spectral losses from Descript's AudioTools
    mel_loss_fn = MelSpectrogramLoss(window_lengths=[2048, 1024, 512, 256, 128], n_mels = [320, 160, 80, 40, 20], mel_fmin=[0,0,0,0,0], mel_fmax=[None,None,None,None,None])
    full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[2048, 1024, 512, 256, 128, 64])
    mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16])

    # writer = SummaryWriter()

    try:
        os.mkdir(TEST_OUT)
    except FileExistsError:
        pass

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, x in loop:
            # forward
            x = x.to(DEVICE)
            x_hat, mu, sigma = model(x)
            x_hat = x_hat[:,:,:block_length] # the model works on frames of length (strides x pqmf_bands) = 4x4x4x2x16 = 2048 samples

            # losses
            x_hat_AS, x_AS = AudioSignal(x_hat, SAMPLE_RATE), AudioSignal(x, SAMPLE_RATE)
            fullband_reconstruction_loss = mel_loss_fn(x_hat_AS, x_AS) + full_stft_loss_fn(x_hat_AS, x_AS)

            multiband_reconstruction_loss = mb_stft_loss_fn(AudioSignal(pqmf(x_hat), SAMPLE_RATE), AudioSignal(pqmf(x), SAMPLE_RATE))
            reconstruction_loss = fullband_reconstruction_loss + multiband_reconstruction_loss

            kl_div = -torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # backprop
            step_nr = epoch*len(train_loader) + i
            # beta = cos_annealed_beta(step_nr, NUM_EPOCHS*len(train_loader))
            beta = 1
            loss = reconstruction_loss + beta*kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({"loss": loss.item(),
                            "recon_loss": reconstruction_loss.item(), 
                            "kl_div": kl_div.item(), 
                            "beta": beta})
        # writer.add_scalar("Loss/train", loss, epoch)
        # writer.add_scalar("Learning_rate", scheduler.get_last_lr(), epoch)
        # scheduler.step()
        if epoch % TEST_FREQ == 0 and TEST_FILE is not None:
            with torch.inference_mode():
                test_wave = [load_mono(TEST_FILE, SAMPLE_RATE)]
                test_segs = reshape_data(test_wave, block_length).to(DEVICE)
                reconstructed_wave = torch.cat([model(seg.unsqueeze(0))[0][:,:,:block_length] for seg in test_segs], dim=2).squeeze(0)
                torchaudio.save(f"{TEST_OUT}/epoch_{epoch}.wav", reconstructed_wave.cpu(), SAMPLE_RATE)

    if TEST_FILE is not None:
        with torch.inference_mode():
            test_wave = [load_mono(TEST_FILE, SAMPLE_RATE)]
            test_segs = reshape_data(test_wave, block_length).to(DEVICE)
            reconstructed_wave = torch.cat([model(seg.unsqueeze(0))[0][:,:,:block_length] for seg in test_segs], dim=2).squeeze(0)
            torchaudio.save(f"{TEST_OUT}/epoch_{epoch}.wav", reconstructed_wave.cpu(), SAMPLE_RATE)

        
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train single-instrument VAE model.")

    parser.add_argument("--data", help="Location of data directory", type=str, metavar="path", required=True)
    parser.add_argument("--hdim", help="Number of hidden layer neurons", type=int, metavar="ndims", default=64)
    parser.add_argument("--zdim", help="Number of latent space variables", type=int, metavar="ndims", default=8)
    parser.add_argument("--epochs", help="Max number of training epochs", type=int, metavar="epochs", default=100)
    parser.add_argument("--batchsize", help="Batch size", type=int, metavar="size", default=32)
    parser.add_argument("--lr", help="Optimizer learning rate", type=float, metavar="rate", default=1e-4)
    parser.add_argument("--bpm", help="Model tempo", metavar="bpm", type=int, default=90)
    parser.add_argument("--subdivs", help="Segments per bar", metavar="divs", type=int, default=8)
    parser.add_argument("--test", help="Test model on audio file after training", type=str, metavar="audio_file_path", default=None)
    parser.add_argument("--out", help="Name of output test file dir", type=str, metavar="dirname", default=None)
    parser.add_argument("--outfreq", help="How often to generate test outputs (once every <epochs>)", type=int, metavar="epochs", default=5)
    args=parser.parse_args()
    main(args)
