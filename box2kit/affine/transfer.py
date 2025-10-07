import torch
import torchaudio
import dac
import argparse
import pickle as pkl
import numpy as np

from box2kit.utils import load_data as uload
from box2kit.affine.model import AffineTransfer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    model_path = args.ckpt

    input_dir = args.ins
    output_dir = args.output

    requantize = args.rq

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    model_sr = codec.sample_rate

    with open(model_path, "rb") as file:
        gen_model = pkl.load(file)
    
    # segment_length_in_samples = int(model_sr*4/(SUBDIV*TEMPO))

    input_waves, file_names = uload.load_dir(input_dir, model_sr)

    uload.mkdir(output_dir)
    
    with torch.inference_mode():
        for input_wave, file_name in zip(input_waves, file_names):
            input_vecs = codec.encode(input_wave.to(DEVICE).unsqueeze(0))[0].transpose(1,2).reshape(-1,1024).cpu().numpy()
            transformed_vecs = gen_model(input_vecs)
            transformed_latents = torch.tensor(np.transpose(transformed_vecs)).reshape(1, 1024, -1).to(DEVICE)

            if requantize:
                transformed_latents = codec.quantizer(transformed_latents)[0]

            output_wave = codec.decode(transformed_latents).reshape(1,-1)

            torchaudio.save(f"{output_dir}/{file_name}", output_wave.detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transfer input recordings using pre-calculated affine transformation.")

    parser.add_argument("ckpt", help="Path to trained affine transformation.", type=str, metavar="ckpt_path", required=True)
    parser.add_argument("ins", help="Location of input audio files.", type=str, metavar="input_path", required=True)
    parser.add_argument("--output", help="Output directory.", type=str, metavar="path", required=True)
    parser.add_argument("--rq", help="Requantize generated latent sequence", action="store_true")
    args = parser.parse_args()
    main(args)