import dac
import torchaudio
from box2kit.autoconcat.model import PairedCodebook, MatchSearchTransfer
import torch
import argparse
import os

import box2kit.utils.load_data as uload

def transfer(file: str, order: int, codec: dac.DAC, gen_model: MatchSearchTransfer):
    sr = codec.sample_rate
    waveform_segment_length = gen_model.codebook.waveform_segment_length

    input_wave_segments = uload.reshape_data([uload.load_mono(file, sr)], waveform_segment_length)

    input_latents = uload.safe_encode(input_wave_segments, codec)
    output_latents = gen_model.transfer_sequence(input_latents, order)
    output_wave_segments = uload.safe_decode(output_latents, codec)

    output_wave = output_wave_segments.reshape(1, -1)
    return output_wave

def main(args):
    EXPERIMENT_NAME = args.name
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    ORDER = args.order

    file_names = [file for file in sorted(os.listdir(INPUT_DIR)) if file[-4:] == ".wav"]

    with torch.inference_mode():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codec = dac.DAC.load(dac.utils.download()).to(device)

        model_sr = codec.sample_rate

        gen_model = torch.load(EXPERIMENT_NAME, map_location=device)

        for file in file_names:
            output = transfer(file, codec, ORDER, gen_model)
            torchaudio.save(f"{OUTPUT_DIR}/{INPUT_DIR}/{file}", output.detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-generated codebook.")

    parser.add_argument("--name", help="File name of codebook.", type=str, metavar="name", required=True)
    parser.add_argument("--input", help="Location of input file directory.", type=str, metavar="path", required=True)
    parser.add_argument("--output", help="Location of output file directory.", type=str, metavar="path", default=uload.mkdir("outs"))
    parser.add_argument("--order", help="Match search order.", type=int, metavar="order", default=1)
    args=parser.parse_args()
    main(args)
