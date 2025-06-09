import torch
from tqdm import tqdm
from torch import nn, optim
from svae.model import SingleVAE, WaveSegmentDataset, PQMFVAE, GenerationCallback, TransferVAE, PairedWaveformDataset
import torchaudio
from torch.utils.data import DataLoader
from utils.load_data import load_dir, load_mono, reshape_data
from utils.checkpoints import get_checkpoint_path
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from audiotools import AudioSignal
from torch.utils.tensorboard import SummaryWriter
import argparse
import math
import os
from svae.rave_pqmf import PQMF
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 48000

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
    CKPT_LOAD = args.loadckpt
    SORT_KEY = args.ckptkey
    DESCENDING = not args.asc
    EXPERIMENT_NAME = args.name

    block_length = int(SAMPLE_RATE*60/(TEMPO*SUBDIVS/4))
    # trunc_block_length = (block_length//2048)*2048
    # remainder = block_length % 2048


    query_dir = args.query
    target_dir = args.target

    dataset = PairedWaveformDataset(query_dir, target_dir, block_length, SAMPLE_RATE)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model = SingleVAE(block_length, H_DIM, Z_DIM).to(DEVICE)
    # model = PQMFVAE(pqmf).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=0.0, last_epoch=-1)

    # # spectral losses from Descript's AudioTools
    # mel_loss_fn = MelSpectrogramLoss(window_lengths=[4096, 2048, 1024, 512, 256], n_mels = [320, 160, 80, 40, 20], mel_fmin=[0,0,0,0,0], mel_fmax=[None,None,None,None,None])
    # full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[1024, 512, 256, 128, 64, 32])
    # mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16])

    model = TransferVAE(block_length, lr=LR)

    # load from previously saved checkpoint, if provided
    ckpt = get_checkpoint_path(CKPT_LOAD, SORT_KEY, DESCENDING) if CKPT_LOAD is not None else None

    tblogger = TensorBoardLogger(save_dir="transvae_logs", name=EXPERIMENT_NAME)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=NUM_EPOCHS, callbacks=GenerationCallback(block_length, TEST_FILE ,TEST_OUT, TEST_FREQ), logger=tblogger)
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=ckpt)

        
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train single-instrument VAE model.")

    parser.add_argument("--query", help="Location of query data.", type=str, metavar="path", required=True)
    parser.add_argument("--target", help="Location of target data.", type=str, metavar="path", required=True)
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
    parser.add_argument("--loadckpt", help="Resume training from checkpoint in lightning_logs folder.", type=str, metavar="checkpoint_folder_path", default=None)
    parser.add_argument("--ckptkey", help="Sorting key for checkpoint in folder.", type=str, metavar="key", default="step")
    parser.add_argument("--asc", help="Sort checkpoints according to key in ascending order.", action="store_true")
    parser.add_argument("--name", help="Name of experiment.", type=str, metavar="experiment_name", default="lightning_logs")
    args=parser.parse_args()
    main(args)
