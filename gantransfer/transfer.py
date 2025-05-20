import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import datetime

import dac
import torchaudio
from ganmodel import Generator, Discriminator, PairedWaveformDataset, DACGAN
from train import load_mono

def main(args):
    input_path = args.input
    chkpt_path = args.chkpt
    output_file_name = args.out
    requantize = args.requantize

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # gen_model = Generator().to(device)
    # discr_model = Discriminator().to(device)
    # # gen_model.load_state_dict(save_state["gen_model"])

    # # block_length_in_samples = save_state["block_length_in_samples"]

    # # gen_model.eval()
    # dac_model = dac.DAC.load(dac.utils.download()).to(device)
    # model_sr = dac_model.sample_rate

    gan = DACGAN.load_from_checkpoint(chkpt_path, Generator().to(device), Discriminator().to(device), dac.DAC.load(dac.utils.download()).to(device), device, 1, 1)

    gen_model = gan.generator
    dac_model = gan.codec

    block_length_in_samples = gan.block_length_in_samples
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
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained generator model.\n")
    # "File pairs must have the same names within their respective directories.\n"
    # "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--input", help="Path to input audio file.", type=str, metavar="path", required=True)
    parser.add_argument("--chkpt", help="Path to model checkpoint.", type=str, metavar="path", required=True)
    parser.add_argument("--out", help="Name of output file.", type=str, metavar="name", required=True)
    parser.add_argument("--requantize", help="Requantize embeddings after applying transformation using DAC's RVQ.", action="store_true")
    args=parser.parse_args()
    main(args)