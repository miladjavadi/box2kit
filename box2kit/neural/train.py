import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

import dac
import torchaudio
from box2kit.neural.model import DACGANV2, GenerationCallback
from box2kit.utils.callbacks import DelayedEarlyStopping
from box2kit.utils.load_data import mkdir, load_configs, reshape_data, PairedWaveformDataset
from box2kit.utils.checkpoints import get_checkpoint_path

# Load audio file
def load_mono(file_name: str, target_sr: int) -> torch.Tensor:
    audio, sr = torchaudio.load(file_name)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    audio = audio.clamp(-1, 1)
    return audio


def prepare_dataloader(query_dir: str, target_dir: str, block_length_in_samples: int, batch_size: int, model_sr: int, device: str) -> torch.utils.data.DataLoader:
    paired_dataset = PairedWaveformDataset(query_dir, target_dir, block_length_in_samples, model_sr)
    dataloader = torch.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NFFT = 1024
test_output_folder = "test_out"

def main(args, configs):
    global_config = configs["global"]
    model_config = configs["neural"]

    # configurables
    tempo = model_config["tempo"]
    subdiv = model_config["subdiv"]

    max_epochs = model_config["epochs"]
    batch_size = model_config["batch_size"]
    lr = model_config["lr"]
    warmup = model_config["warmup"]
    es_delay = model_config["es_delay"]
    beta = model_config["beta"]
    phi = model_config["phi"]

    models_dir = mkdir(global_config["models"])
    logs_dir = model_config["logs"]
    test_freq = model_config["test_freq"]

    train_target_path = global_config["training_target_path"]
    train_output_path = global_config["training_output_path"]
    val_target_path = global_config["validation_target_path"]
    val_output_path = global_config["validation_output_path"]

    # command-line arguments
    data_path = args.data
    experiment_name = args.name
    ckpt = args.ckpt
    test_file = args.test

    train_target_dir = os.path.join(data_path, train_target_path)
    train_output_dir = os.path.join(data_path, train_output_path)
    val_target_dir = os.path.join(data_path, val_target_path)
    val_output_dir = os.path.join(data_path, val_output_path)

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE) # type: ignore
    sample_rate = codec.sample_rate
    block_length_in_samples = int(sample_rate*60/(tempo*subdiv/4))

    train_loader = prepare_dataloader(train_target_dir, train_output_dir, block_length_in_samples, batch_size, sample_rate, DEVICE)

    if os.path.exists(val_target_dir) and os.path.exists(val_output_dir):
        val_loader = prepare_dataloader(val_target_dir, val_output_dir, block_length_in_samples, batch_size, sample_rate, DEVICE)
    else:
        print("No validation data found, skipping model validation.")
        val_loader = None

    # the length of an audio block may be altered during decoding.
    # thus, a second block sample length must be passed to the discriminator
    with torch.inference_mode():
        dummy_frame = codec.encode(train_loader.dataset[0][0].unsqueeze(0).to(DEVICE))[0]
        block_length_in_frames = dummy_frame.shape[2]
        output_block = codec.decode(dummy_frame)
        output_block_length_in_samples = output_block.shape[2]
        dummy_stft = torch.stft(output_block.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=output_block.device)).abs()

    # gan = DACGAN(dac_model, device, block_length_in_samples, output_block_length_in_samples, block_length_in_frames, lambda_embedding=lambda_embedding) # initialize new model
    gan = DACGANV2(block_length_in_samples, output_block_length_in_samples, block_length_in_frames, [dummy_stft.shape[1], dummy_stft.shape[2]], NFFT, beta, phi, codec, warmup=warmup, lr=lr)

    callbacks = []

    if test_file is not None:
        callbacks.append(GenerationCallback(test_file, test_output_folder, test_freq))
    if val_loader is not None:
        callbacks.extend([DelayedEarlyStopping(es_delay, monitor="val_loss", mode="min", patience=10, check_finite=True), ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best-{epoch:02d}-{val_loss:.2f}", save_last=True)])


    # load from previously saved checkpoint, if provided
    # ckpt = get_checkpoint_path(ckpt_load, "step", True) if ckpt_load is not None else None

    logger = CSVLogger(save_dir=os.path.join(models_dir, logs_dir), name=experiment_name)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=max_epochs, logger=logger, callbacks=callbacks) # type: ignore
    trainer.fit(gan, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt)


if __name__ == "__main__":
    configs = load_configs("box2kit/configs")
    parser=argparse.ArgumentParser(description="Train neural timbre transfer model using paired query/target datasets.\n"
    "File pairs must have the same names within their respective directories.\n"
    "For instance: <target>/x.wav should have a corresponding <out>/x.wav.")
    
    parser=argparse.ArgumentParser(description="Train paired instrument VAE model.")
    parser.add_argument("data", help="Location of training and validaiton data.", type=str, metavar="path")
    parser.add_argument("--ckpt", help="Resume training from checkpoint in a log folder.", type=str, metavar="logs_path", default=None)
    parser.add_argument("--test", help="While training, periodically test model on audio file.", type=str, metavar="audio_file_path", default=None)
    parser.add_argument("--name", help="Name of experiment", type=str, metavar="name", default="default_logs")
    
    args=parser.parse_args()
    main(args, configs)
