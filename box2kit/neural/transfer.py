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
from box2kit.neural.model import DACGANV2
from box2kit.utils import load_data as uload

def main(args):
    input_dir = args.ins
    ckpt = args.ckpt
    output_dir = args.out
    requantize = args.rq

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gan = DACGANV2.load_from_checkpoint(ckpt, map_location=device) # type: ignore

    gen_model = gan.generator
    dac_model = gan.codec

    block_length_in_samples = gan.input_block_length
    model_sr = dac_model.sample_rate

    input_waveforms, file_names = uload.load_dir(input_dir, model_sr)
    uload.mkdir(output_dir)

    for input_waveform, file_name in zip(input_waveforms, file_names):
        # trim waveform to whole number of block lengths
        input_waveform = input_waveform[:,:((input_waveform.shape[1]//block_length_in_samples)*block_length_in_samples)].to(device)

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

        torchaudio.save(f"{output_dir}/{file_name}", reconstructed_waveform.unsqueeze(0).detach().cpu(), model_sr)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained neural transfer model.")

    parser.add_argument("ins", help="Path to input audio folder.", type=str, metavar="input_path", required=True)
    parser.add_argument("ckpt", help="Path to checkpoint.", type=str, metavar="ckpt_path", required=True)
    parser.add_argument("--out", help="Name of output folder.", type=str, metavar="name", required=True)
    parser.add_argument("--rq", help="Requantize embeddings after applying transformation using DAC's RVQ.", action="store_true")
    args=parser.parse_args()
    main(args)