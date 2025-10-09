import torchaudio
from box2kit.waveenc.model import TransferGAN
import torch
import argparse
import os

import box2kit.utils.load_data as uload
from box2kit.utils import checkpoints

def transfer(input_wave: torch.Tensor, gen_model: TransferGAN) -> torch.Tensor:
    output_wave = gen_model(input_wave)
    return output_wave

def main(args):
    ckpt = args.ckpt
    input_dir = args.ins
    output_dir = args.out

    device = "cuda" if torch.cuda.is_available() else "cpu"

    uload.mkdir(output_dir)

    # ckpt = checkpoints.get_checkpoint_path(ckpt, sort_key, descending)

    gen_model = TransferGAN.load_from_checkpoint(ckpt, map_location=device)
    gen_model.eval()
    # model_sr = gen_model.sample_rate
    model_sr = 44100

    input_waveforms, file_names = uload.load_dir(input_dir, model_sr)

    with torch.inference_mode():
        for input_wave, file_name in zip(input_waveforms, file_names):
            input_wave = input_wave.to(device)
            output, _, _ = transfer(input_wave.reshape(1,1,-1), gen_model)
            torchaudio.save(f"{output_dir}/{file_name}", output[0].detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-generated codebook.")

    parser.add_argument("ins", help="Location of input file directory.", type=str, metavar="input_path", required=True)
    parser.add_argument("ckpt", help="Path to checkpoint.", type=str, metavar="ckpt_path", required=True)
    parser.add_argument("--out", help="Location of output file directory.", type=str, metavar="output_path", default=uload.mkdir("outs"))
    args=parser.parse_args()
    main(args)
