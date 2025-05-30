import torch
from tqdm import tqdm
from torch import nn, optim
from svae.model import SingleVAE, WaveSegmentDataset
import torchaudio
from torch.utils.data import DataLoader
# from utils.load_data import load_dir, load_mono
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from audiotools import AudioSignal
import argparse

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 48000

def annealed_beta(current_step, total_steps):
    return min(1.0, current_step / total_steps)

def main(args):
    H_DIM = args.hdim
    Z_DIM = args.zdim
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    LR = args.lr
    TEMPO = args.bpm
    SUBDIVS = args.subdivs

    block_length = int(SAMPLE_RATE*60/(TEMPO*SUBDIVS/4))

    dataset_path = args.data

    dataset = WaveSegmentDataset(dataset_path, block_length, SAMPLE_RATE)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SingleVAE(block_length, H_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mel_loss_fn = MelSpectrogramLoss()
    stft_loss_fn = MultiScaleSTFTLoss()

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, x in loop:
            # forward
            x = x.to(DEVICE)
            x_hat, mu, sigma = model(x)

            # losses
            x_hat_AS, x_AS = AudioSignal(x_hat, SAMPLE_RATE), AudioSignal(x, SAMPLE_RATE)
            reconstruction_loss = mel_loss_fn(x_hat_AS, x_AS) + stft_loss_fn(x_hat_AS, x_AS)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # backprop
            step_nr = epoch*len(train_loader) + i
            beta = annealed_beta(step_nr, 10000)
            loss = reconstruction_loss + beta*kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({"loss": loss.item(),
                            "recon_loss": reconstruction_loss.item(), 
                            "kl_div": kl_div.item()})
        
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train single-instrument VAE model.")

    parser.add_argument("--data", help="Location of data directory", type=str, metavar="path", required=True)
    parser.add_argument("--hdim", help="Number of hidden layer neurons", type=int, metavar="ndims", default=200)
    parser.add_argument("--zdim", help="Number of latent space variables", type=int, metavar="ndims", default=8)
    parser.add_argument("--epochs", help="Max number of training epochs", type=int, metavar="epochs", default=100)
    parser.add_argument("--batchsize", help="Batch size", type=int, metavar="size", default=32)
    parser.add_argument("--lr", help="Optimizer learning rate", type=float, metavar="rate", default=3e-5)
    parser.add_argument("--bpm", help="Model tempo", metavar="bpm", default=90)
    parser.add_argument("--subdivs", help="Segments per bar", metavar="divs", default=8)
    args=parser.parse_args()
    main(args)
