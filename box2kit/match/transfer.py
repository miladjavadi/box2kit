import dac
import torchaudio
from box2kit.match.model import PairedCodebook, MatchSearchTransfer
import torch
import argparse
import os

import box2kit.utils.load_data as uload

def transfer(dir: str, file: str, order: int, codec: dac.DAC, gen_model: MatchSearchTransfer):
    sr = codec.sample_rate
    waveform_segment_length = gen_model.codebook.waveform_segment_length

    input_wave_segments = uload.reshape_data([uload.load_mono(f"{dir}/{file}", sr).to(codec.device)], waveform_segment_length)

    input_latents = uload.safe_encode(input_wave_segments, codec)
    output_latents = gen_model.transfer_sequence(input_latents, order)
    output_wave_segments = uload.safe_decode(output_latents, codec)

    output_wave = output_wave_segments.reshape(1, -1)
    return output_wave

def main(args):
    experiment_name = args.name
    input_dir = args.ins
    output_dir = args.output
    order = args.order

    uload.mkdir(f"{output_dir}")

    file_names = [file for file in sorted(os.listdir(input_dir)) if file[-4:] == ".wav"]

    with torch.inference_mode():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codec = dac.DAC.load(dac.utils.download()).to(device)

        model_sr = codec.sample_rate

        with torch.serialization.safe_globals([PairedCodebook]):
            gen_model = MatchSearchTransfer(torch.load(experiment_name, map_location=device))

        for file in file_names:
            output = transfer(input_dir, file, order, codec, gen_model)
            torchaudio.save(f"{output_dir}/{file}", output.detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-generated codebook.")

    parser.add_argument("name", help="File name of codebook.", type=str, metavar="coebook_name")
    parser.add_argument("ins", help="Location of input file directory.", type=str, metavar="input_path")
    parser.add_argument("--output", help="Location of output file directory.", type=str, metavar="output_path", default=uload.mkdir("outs"))
    parser.add_argument("--order", help="Match search order.", type=int, metavar="order", default=1)
    args=parser.parse_args()
    main(args)
