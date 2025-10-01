import torchaudio
from box2kit.svae.model import TransferGAN
import torch
import argparse
import os

import box2kit.utils.load_data as uload
from box2kit.utils import checkpoints

def transfer(input_wave: torch.Tensor, gen_model: TransferGAN) -> torch.Tensor:
    output_wave = gen_model(input_wave)
    return output_wave

def main(args):
    CKPT_DIR = args.ckpt
    SORT_KEY = args.key
    DESCENDING = not args.asc
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output

    device = "cuda" if torch.cuda.is_available() else "cpu"

    uload.mkdir(OUTPUT_DIR)

    ckpt = checkpoints.get_checkpoint_path(CKPT_DIR, SORT_KEY, DESCENDING)

    gen_model = TransferGAN.load_from_checkpoint(ckpt)
    gen_model.eval()
    # model_sr = gen_model.sample_rate
    model_sr = 44100

    input_waveforms, file_names = uload.load_dir(INPUT_DIR, model_sr)

    with torch.inference_mode():
        for input_wave, file_name in zip(input_waveforms, file_names):
            input_wave.to(device)
            output = transfer(input_wave.reshape(1,1,-1), gen_model)
            torchaudio.save(f"{OUTPUT_DIR}/{file_name}", output.detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-generated codebook.")

    parser.add_argument("--ckpt", help="Location of checkpoint directory.", type=str, metavar="name", required=True)
    parser.add_argument("--key", help="Sorting key for checkpoint in folder.", type=str, metavar="key", default="step")
    parser.add_argument("--asc", help="Sort checkpoints according to key in ascending order.", action="store_true")
    parser.add_argument("--input", help="Location of input file directory.", type=str, metavar="path", required=True)
    parser.add_argument("--output", help="Location of output file directory.", type=str, metavar="path", default=uload.mkdir("outs"))
    args=parser.parse_args()
    main(args)
