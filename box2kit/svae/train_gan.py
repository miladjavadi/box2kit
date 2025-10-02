import torch
from tqdm import tqdm
from torch import nn, optim
from box2kit.svae.model import SingleVAE, WaveSegmentDataset, PQMFVAE, GenerationCallback, TransferGAN, PairedWaveformDataset
import torchaudio
from torch.utils.data import DataLoader
from box2kit.utils.load_data import load_dir, load_mono, reshape_data, load_configs
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
from box2kit.utils.callbacks import DelayedEarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

### CURRENT

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NFFT = 1024
TEST_OUT = "test_out"

def main(args, configs):
    global_config = configs["configs"]
    model_config = configs["waveenc"]

    # configurables
    SAMPLE_RATE = model_config["sample_rate"]
    TEMPO = model_config["tempo"]
    SUBDIV = model_config["subdiv"]
    NMOG = model_config["mog"]

    NUM_EPOCHS = model_config["epochs"]
    BATCH_SIZE = model_config["batch_size"]
    LR = model_config["lr"]
    WARMUP = model_config["warmup"]
    ES_DELAY = model_config["es_delay"]
    BETA = model_config["beta"]
    PHI = model_config["phi"]

    MODELS_DIR = global_config["models"]
    LOGS_DIR = model_config["logs"]
    TEST_FREQ = model_config["test_freq"]

    TRAIN_TARGET_PATH = global_config["training_target_path"]
    TRAIN_OUTPUT_PATH = global_config["training_output_path"]
    VAL_TARGET_PATH = global_config["validation_target_path"]
    VAL_OUTPUT_PATH = global_config["validation_output_path"]

    # command-line arguments
    DATA_PATH = args.data
    EXPERIMENT_NAME = args.name
    CKPT_LOAD = args.ckpt
    TEST_FILE = args.test

    block_length = int(SAMPLE_RATE*60/(TEMPO*SUBDIV/4))
    # trunc_block_length = (block_length//2048)*2048
    # remainder = block_length % 2048

    train_target_dir = f"{DATA_PATH}/{TRAIN_TARGET_PATH}"
    train_output_dir = f"{DATA_PATH}/{TRAIN_OUTPUT_PATH}"
    val_target_dir = f"{DATA_PATH}/{VAL_TARGET_PATH}"
    val_output_dir = f"{DATA_PATH}/{VAL_OUTPUT_PATH}"

    train_dataset = PairedWaveformDataset(train_target_dir, train_output_dir, block_length, SAMPLE_RATE)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if os.path.exists(val_target_dir) and os.path.exists(val_output_dir):
        val_dataset = PairedWaveformDataset(val_target_dir, val_output_dir, block_length, SAMPLE_RATE)
        val_loader = DataLoader(dataset=val_dataset, batch_size = BATCH_SIZE, shuffle=True)
    else:
        print("No validation data found, skipping model validation.")
        val_loader = None

    # model = SingleVAE(block_length, H_DIM, Z_DIM).to(DEVICE)
    # model = PQMFVAE(pqmf).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=0.0, last_epoch=-1)

    # # spectral losses from Descript's AudioTools
    # mel_loss_fn = MelSpectrogramLoss(window_lengths=[4096, 2048, 1024, 512, 256], n_mels = [320, 160, 80, 40, 20], mel_fmin=[0,0,0,0,0], mel_fmax=[None,None,None,None,None])
    # full_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[1024, 512, 256, 128, 64, 32])
    # mb_stft_loss_fn = MultiScaleSTFTLoss(window_lengths=[128, 64, 32, 16])

    with torch.inference_mode():
        dummy = train_dataset[0][0]
        dummy_stft = torch.stft(dummy.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=dummy.device)).abs()

    model = TransferGAN(block_length, [dummy_stft.shape[1], dummy_stft.shape[2]], nmog=NMOG, lr=LR, lambda_adversarial=PHI, warmup=WARMUP, beta=BETA)

    # load from previously saved checkpoint, if provided
    ckpt = get_checkpoint_path(CKPT_LOAD, "step", True) if CKPT_LOAD is not None else None

    # checkpoint_monitor = "val_loss" if val_loader is not None else "g_loss"

    callbacks = [GenerationCallback(block_length, TEST_FILE ,TEST_OUT, TEST_FREQ)]

    if val_loader is not None:
        callbacks.extend([DelayedEarlyStopping(ES_DELAY, monitor="val_loss", mode="min", patience=10, check_finite=True), ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")])
    
    logger = CSVLogger(save_dir=os.path.join(MODELS_DIR, LOGS_DIR), name=EXPERIMENT_NAME)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=callbacks) # type: ignore
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt)

        
if __name__ == "__main__":
    configs = load_configs("box2kit/configs")
    print(configs)
    
    parser=argparse.ArgumentParser(description="Train paired instrument VAE model.")
    parser.add_argument("data", help="Location of training and validaiton data.", type=str, metavar="path", required=True)
    parser.add_argument("--ckpt", help="Resume training from checkpoint in a log folder.", type=str, metavar="logs_path", default=None)
    parser.add_argument("--test", help="While training, periodically test model on audio file.", type=str, metavar="audio_file_path", default=None)
    
    args=parser.parse_args()
    main(args, configs)