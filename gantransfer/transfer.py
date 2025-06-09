import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import yaml

import dac
import torchaudio
from gantransfer.ganmodel import Generator, Discriminator, PairedWaveformDataset, DACGAN
from gantransfer.train import load_mono, load_checkpoint

def main(args):
    input_path = args.input
    ckpt_folder = args.ckpt
    output_file_name = args.out
    requantize = args.requantize
    key = args.key
    descending = not args.asc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gan = load_checkpoint(ckpt_folder, dac.DAC.load(dac.utils.download()).to(device), device, key=key, descending=descending)

    gen_model = gan.generator
    dac_model = gan.codec

    block_length_in_samples = gan.input_block_length
    model_sr = dac_model.sample_rate

    input_waveform = load_mono(input_path, model_sr).to(device)
    # trim waveform to whole number of block lengths
    input_waveform = input_waveform[:,:((input_waveform.shape[1]//block_length_in_samples)*block_length_in_samples)]

    # reshape waveform into blocks
    input_blocks = torch.reshape(input_waveform, (-1, block_length_in_samples))
    input_blocks = input_blocks.unsqueeze(1)

    with torch.inference_mode():
        input_embeddings = dac_model.encode(input_blocks)[0]

        transformed_embeddings = gen_model(input_embeddings)

        if requantize:
            transformed_embeddings = dac_model.quantizer(transformed_embeddings, None)[0]

        reconstruction = dac_model.decode(transformed_embeddings)

    reconstructed_waveform = reconstruction.flatten()

    torchaudio.save(output_file_name, reconstructed_waveform.unsqueeze(0).detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained generator model.")
    # parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained generator model.\n"
    # "File pairs must have the same names within their respective directories.\n"
    # "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--input", help="Path to input audio file.", type=str, metavar="path", required=True)
    parser.add_argument("--ckpt", help="Path to checkpoint folder.", type=str, metavar="path", required=True)
    parser.add_argument("--key", help="Sorting key for checkpoint in folder.", type=str, metavar="key", default="step")
    parser.add_argument("--asc", help="Sort checkpoints according to key in ascending order.", action="store_true")
    parser.add_argument("--out", help="Name of output file.", type=str, metavar="name", required=True)
    parser.add_argument("--requantize", help="Requantize embeddings after applying transformation using DAC's RVQ.", action="store_true")
    args=parser.parse_args()
    main(args)