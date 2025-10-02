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
from box2kit.gantransfer.ganmodel import Generator, Discriminator, PairedWaveformDataset, DACGAN, DACGANV2
from box2kit.utils.callbacks import DelayedEarlyStopping
from box2kit.utils.load_data import mkdir, load_configs

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

def reshape_dataset(waveforms: list[torch.Tensor], block_length_in_samples: int) -> torch.Tensor:
    # List([1 x waveform_length]) -> [n_blocks x 1 x block_lengths_in_samples]

    dataset = torch.zeros((0, block_length_in_samples), device=waveforms[0].device)
    for waveform in waveforms:
        # trim waveform to whole number of block lengths
        waveform = waveform[:,:((waveform.shape[1]//block_length_in_samples)*block_length_in_samples)]

        # reshape waveform into blocks
        blocks = torch.reshape(waveform, (-1, block_length_in_samples))
        dataset = torch.cat((dataset, blocks), dim=0)
    
    dataset = dataset.unsqueeze(1)
    return dataset

def prepare_dataloader(query_dir: str, target_dir: str, block_length_in_samples: int, batch_size: int, model_sr: int, device: str) -> torch.utils.data.DataLoader:
    query_files = sorted(os.listdir(query_dir))
    target_files = sorted(os.listdir(target_dir))

    # load in all waveforms
    query_waveforms = [load_mono((f"{query_dir}/{file}"), model_sr).to(device) for file in query_files if file[-4:] == ".wav"]
    target_waveforms = [load_mono((f"{target_dir}/{file}"), model_sr).to(device) for file in target_files if file[-4:] == ".wav"]

    query_dataset = reshape_dataset(query_waveforms, block_length_in_samples)
    target_dataset = reshape_dataset(target_waveforms, block_length_in_samples)
    paired_dataset = PairedWaveformDataset(query_dataset, target_dataset)
    dataloader = torch.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# deprecated in favor of lightning-embedded training
def training_procedure(gen_model, discr_model, dac_model, dataloader, epochs, device) -> None:
    embedding_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCELoss()
    lambda_embedding = 100

    gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.00002, betas=(0.5, 0.999))
    discr_optimizer = optim.Adam(discr_model.parameters(), lr=0.00002, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    for i in range(epochs):
        for batch_nr, (query, target) in enumerate(dataloader):
            with torch.no_grad():
                Z_query = dac_model.encode(query)[0]
                Z_target = dac_model.encode(target)[0]

            # train discriminator
            Z_transformed = gen_model(Z_query).detach()
            
            with torch.no_grad():
                transformed_decoded = dac_model.decode(Z_transformed)

            target = target[:,:,:transformed_decoded.shape[2]] # trim tail of target that is lost when decoding
            
            d_real = discr_model(Z_query, target)
            d_fake = discr_model(Z_query, transformed_decoded)

            real_labels = torch.full(d_real.shape, real_label, device=device, dtype=torch.float32)
            real_adversarial_loss = adversarial_loss_fn(d_real, real_labels)

            fake_labels = torch.full(d_fake.shape, fake_label, device=device, dtype=torch.float32)
            fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

            discr_loss = real_adversarial_loss + fake_adversarial_loss
            discr_optimizer.zero_grad()
            discr_loss.backward(retain_graph=True)
            discr_optimizer.step()

            # train generator
            
            Z_transformed = gen_model(Z_query)
            embedding_loss = embedding_loss_fn(Z_transformed, Z_target)

            d_fake = discr_model(Z_query, transformed_decoded)
            fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

            gen_loss = 1/fake_adversarial_loss + lambda_embedding * embedding_loss
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

def get_checkpoint_path(checkpoint_folder: str, key: str = "step", descending: bool = True) -> str:
    checkpoint_files = [file for file in os.listdir(f"{checkpoint_folder}/checkpoints") if file[-5:] == ".ckpt"]

    checkpoints = [dict([["name", name]] + [attribute.split("=") for attribute in name.split("-")]) for name in checkpoint_files]
    
    try:
        checkpoint_name = sorted(checkpoints, key = lambda d: d[key], reverse = descending)[0]["name"]
    
    except KeyError:
        raise KeyError(f'Key "{key}" not found in checkpoint file name.')
    
    checkpoint_path = f"{checkpoint_folder}/checkpoints/{checkpoint_name}"
    return checkpoint_path

def load_checkpoint(checkpoint_folder: str, codec, device: str, key: str = "step", descending: bool = True):
    hparams_file = f"{checkpoint_folder}/hparams.yaml"

    with open(hparams_file) as f:
        hparams = yaml.safe_load(f)

    checkpoint_path = get_checkpoint_path(checkpoint_folder, key, descending=descending)

    # checkpoint = DACGAN.load_from_checkpoint(checkpoint_path, codec=codec, device=device, **hparams)
    checkpoint = DACGANV2.load_from_checkpoint(checkpoint_path, map_location=torch.device(device), **hparams)

    return checkpoint


def main(args):
    target_dir = args.target
    output_dir = args.out
    val_target_dir = args.vtarget
    val_output_dir = args.voutput

    tempo = args.tempo
    subdivs = args.subdiv
    batch_size = args.batchsize
    max_epochs = args.maxepochs
    ckpt_load = args.loadckpt
    lambda_embedding = args.lemb
    lambda_adversarial = args.ladv
    warmup = args.warmup
    sort_key = args.ckptkey
    descending = not args.asc
    experiment_name = args.name
    nfft = 1024

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dac_model = dac.DAC.load(dac.utils.download()).to(device) # type: ignore
    model_sr = dac_model.sample_rate
    block_length_in_samples = int(model_sr*60/(tempo*subdivs/4))

    train_dataloader = prepare_dataloader(target_dir, output_dir, block_length_in_samples, batch_size, model_sr, device)

    if val_target_dir is None or val_output_dir is None:
        val_loader = None
    else:
        val_loader = prepare_dataloader(val_target_dir, val_output_dir, block_length_in_samples, batch_size, model_sr, device)

    # the length of an audio block may be altered during decoding.
    # thus, a second block sample length must be passed to the discriminator
    with torch.inference_mode():
        dummy_frame = dac_model.encode(train_dataloader.dataset[0][0].unsqueeze(0))[0]
        block_length_in_frames = dummy_frame.shape[2]
        output_block = dac_model.decode(dummy_frame)
        output_block_length_in_samples = output_block.shape[2]
        dummy_stft = torch.stft(output_block.squeeze(1), nfft, return_complex=True, window=torch.hann_window(nfft, device=output_block.device)).abs()

    # gan = DACGAN(dac_model, device, block_length_in_samples, output_block_length_in_samples, block_length_in_frames, lambda_embedding=lambda_embedding) # initialize new model
    gan = DACGANV2(block_length_in_samples, output_block_length_in_samples, block_length_in_frames, [dummy_stft.shape[1], dummy_stft.shape[2]], nfft, lambda_embedding, lambda_adversarial, dac_model, warmup=warmup)


    if val_loader is not None:
        callbacks = ([DelayedEarlyStopping(int(0.65*max_epochs), monitor="val_loss", mode="min", patience=10, check_finite=True), ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")])
    else:
        callbacks = None
    
    logger = CSVLogger(save_dir="w2w_logs", name=experiment_name)
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=callbacks) # type: ignore
    trainer.fit(gan, train_dataloaders=train_dataloader, val_dataloaders=val_loader, ckpt_path=ckpt)

    # load from previously saved checkpoint, if provided
    ckpt = get_checkpoint_path(ckpt_load, sort_key, descending) if ckpt_load is not None else None

    trainer.fit(gan, train_dataloaders=train_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    configs = load_configs("box2kit/configs")
    parser=argparse.ArgumentParser(description="Train GAN-based timbre transfer model using paired query/carget datasets.\n"
    "File pairs must have the same names within their respective directories.\n"
    "For instance: <target>/x.wav should have a corresponding <out>/x.wav.")
    
    parser=argparse.ArgumentParser(description="Train paired instrument VAE model.")
    parser.add_argument("data", help="Location of training and validaiton data.", type=str, metavar="path")
    parser.add_argument("--ckpt", help="Resume training from checkpoint in a log folder.", type=str, metavar="logs_path", default=None)
    parser.add_argument("--test", help="While training, periodically test model on audio file.", type=str, metavar="audio_file_path", default=None)
    parser.add_argument("--name", help="Name of experiment", type=str, metavar="name", default="default_logs")
    
    args=parser.parse_args()
    main(args, configs)

