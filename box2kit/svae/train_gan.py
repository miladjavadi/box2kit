import torch
from tqdm import tqdm
from torch import nn, optim
from box2kit.svae.model import SingleVAE, WaveSegmentDataset, PQMFVAE, GenerationCallback, TransferGAN, PairedWaveformDataset
import torchaudio
from torch.utils.data import DataLoader
from box2kit.utils.load_data import load_dir, load_mono, reshape_data
from box2kit.utils.checkpoints import get_checkpoint_path
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from audiotools import AudioSignal
from torch.utils.tensorboard import SummaryWriter
import argparse
import math
import os
from box2kit.svae.rave_pqmf import PQMF
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 44100

def main(args):
    H_DIM = args.hdim
    Z_DIM = args.zdim
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    LR = args.lr
    TEMPO = args.bpm
    SUBDIV = args.subdiv
    TEST_FILE = args.test
    TEST_OUT = args.testout
    TEST_FREQ = args.testfreq
    CKPT_LOAD = args.ckpt
    SORT_KEY = args.ckptkey
    DESCENDING = not args.asc
    EXPERIMENT_NAME = args.name
    LAMBDA_ADV = args.ladv
    WARMUP = args.warmup
    NFFT = 1024
    NMOG = args.mog

    block_length = int(SAMPLE_RATE*60/(TEMPO*SUBDIV/4))
    # trunc_block_length = (block_length//2048)*2048
    # remainder = block_length % 2048


    target_dir = args.target
    output_dir = args.output

    dataset = PairedWaveformDataset(target_dir, output_dir, block_length, SAMPLE_RATE)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model = SingleVAE(block_length, H_DIM, Z_DIM).to(DEVICE)
    # model = PQMFVAE(pqmf).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=0.0, last_epoch=-1)

    # # spectral losses from Descript's AudioTools
    # mel_loss_fn = MelSpectrogramLoss(window_lengths=[4096, 2048, 1024, 512, 256], n_mels = [320, 160, 80, 40, 20], mel_fmin=[0,0,0,0,0], mel_fmax=[None,None,None,None,None])
    # full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[1024, 512, 256, 128, 64, 32])
    # mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16])

    with torch.inference_mode():
        dummy = dataset[0][0]
        dummy_stft = torch.stft(dummy.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=dummy.device)).abs()

    model = TransferGAN(block_length, [dummy_stft.shape[1], dummy_stft.shape[2]], nmog=NMOG, lr=LR, lambda_adversarial=LAMBDA_ADV, warmup=WARMUP)

    # load from previously saved checkpoint, if provided
    ckpt = get_checkpoint_path(CKPT_LOAD, SORT_KEY, DESCENDING) if CKPT_LOAD is not None else None

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    logger = CSVLogger(save_dir="neural_logs", name=EXPERIMENT_NAME)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10, check_finite=True), checkpoint_callback, GenerationCallback(block_length, TEST_FILE ,TEST_OUT, TEST_FREQ)]) # type: ignore
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=ckpt)

        
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train single-instrument VAE model.")

    parser.add_argument("--target", help="Location of target data.", type=str, metavar="path", required=True)
    parser.add_argument("--output", help="Location of output data.", type=str, metavar="path", required=True)
    parser.add_argument("--hdim", help="Number of hidden layer neurons", type=int, metavar="ndims", default=64)
    parser.add_argument("--zdim", help="Number of latent space variables", type=int, metavar="ndims", default=8)
    parser.add_argument("--epochs", help="Max number of training epochs", type=int, metavar="epochs", default=100)
    parser.add_argument("--batchsize", help="Batch size", type=int, metavar="size", default=32)
    parser.add_argument("--lr", help="Optimizer learning rate", type=float, metavar="rate", default=1e-4)
    parser.add_argument("--bpm", help="Model tempo", metavar="bpm", type=int, default=90)
    parser.add_argument("--subdiv", help="Segments per bar", metavar="divs", type=int, default=8)
    parser.add_argument("--test", help="Test model on audio file after training", type=str, metavar="audio_file_path", default=None)
    parser.add_argument("--testout", help="Name of output test file dir", type=str, metavar="dirname", default=None)
    parser.add_argument("--testfreq", help="How often to generate test outputs (once every <epochs>)", type=int, metavar="epochs", default=5)
    parser.add_argument("--ckpt", help="Resume training from checkpoint in lightning_logs folder.", type=str, metavar="checkpoint_folder_path", default=None)
    parser.add_argument("--ckptkey", help="Sorting key for checkpoint in folder.", type=str, metavar="key", default="step")
    parser.add_argument("--asc", help="Sort checkpoints according to key in ascending order.", action="store_true")
    parser.add_argument("--name", help="Name of experiment.", type=str, metavar="experiment_name", default="lightning_logs")
    parser.add_argument("--ladv", help="Set importance of adversarial loss in generator cost function.", type=float, metavar="lambda", default=1)
    parser.add_argument("--warmup", help="Number of epochs in warmup phase (no discriminator)", type=int, metavar="epochs", default=250)
    parser.add_argument("--mog", help="Number of Gaussian mixture modes in prior. 0 for standard Gaussian.", type=int, metavar="modes", default=0)
    args=parser.parse_args()
    main(args)
