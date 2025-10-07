import torch
from box2kit.waveenc.model import GenerationCallback, TransferGAN
from torch.utils.data import DataLoader
from box2kit.utils.load_data import load_configs, mkdir, PairedWaveformDataset
from box2kit.utils.checkpoints import get_checkpoint_path
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

### CURRENT

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NFFT = 1024
test_output_folder = "test_out"

def main(args, configs):
    global_config = configs["global"]
    model_config = configs["waveenc"]

    # configurables
    sample_rate = model_config["sample_rate"]
    tempo = model_config["tempo"]
    subdiv = model_config["subdiv"]
    nmog = model_config["mog"]

    max_epochs = model_config["epochs"]
    batch_size = model_config["batch_size"]
    lr = model_config["lr"]
    warmup = model_config["warmup"]
    early_stopping_delay = model_config["es_delay"]
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

    # full-length signals are divided into segments according to tempo
    segment_length = int(sample_rate*60/(tempo*subdiv/4)) 

    train_target_dir = os.path.join(data_path, train_target_path)
    train_output_dir = os.path.join(data_path, train_output_path)
    val_target_dir = os.path.join(data_path, val_target_path)
    val_output_dir = os.path.join(data_path, val_output_path)

    train_dataset = PairedWaveformDataset(train_target_dir, train_output_dir, segment_length, sample_rate)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    if os.path.exists(val_target_dir) and os.path.exists(val_output_dir):
        val_dataset = PairedWaveformDataset(val_target_dir, val_output_dir, segment_length, sample_rate)
        val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size, shuffle=True)
    else:
        print("No validation data found, skipping model validation.")
        val_loader = None

    # calculate stft dims of audio segments to initialize stft discriminator
    with torch.inference_mode():
        dummy = train_dataset[0][0]
        dummy_stft = torch.stft(dummy.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=dummy.device)).abs()

    model = TransferGAN(segment_length, [dummy_stft.shape[1], dummy_stft.shape[2]], nmog=nmog, lr=lr, phi=phi, warmup=warmup, beta=beta)

    # load from previously saved checkpoint, if provided
    # ckpt = get_checkpoint_path(ckpt_load, "step", True) if ckpt_load is not None else None

    # callbacks = [ModelCheckpoint(save_last=True, filename="latest-{epoch:02d}")]
    callbacks = []

    # generate occasional test outputs from provided test file
    if test_file is not None:
        callbacks.append(GenerationCallback(segment_length, test_file, test_output_folder, test_freq))

    # use early stopping and model checkpoints if validation data is provided
    if val_loader is not None:
        callbacks.extend([EarlyStopping(monitor="val_loss", mode="min", patience=100, check_finite=True), ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best-{epoch:02d}-{val_loss:.2f}", save_last=True)])
    
    logger = CSVLogger(save_dir=os.path.join(models_dir, logs_dir), name=experiment_name)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=max_epochs, logger=logger, callbacks=callbacks, min_epochs=early_stopping_delay) # type: ignore
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt)

        
if __name__ == "__main__":
    configs = load_configs("box2kit/configs")
    
    parser=argparse.ArgumentParser(description="Train paired instrument VAE model.")
    parser.add_argument("data", help="Location of training and validaiton data.", type=str, metavar="path")
    parser.add_argument("--ckpt", help="Resume training from checkpoint in a log folder.", type=str, metavar="logs_path", default=None)
    parser.add_argument("--test", help="While training, periodically test model on audio file.", type=str, metavar="audio_file_path", default=None)
    parser.add_argument("--name", help="Name of experiment", type=str, metavar="name", default="default_logs")
    
    args=parser.parse_args()
    main(args, configs)