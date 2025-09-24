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
    # TEMPO = args.tempo
    # SUBDIV = args.subdiv
    # BATCH_SIZE = args.batchsize
    MODEL_PATH = args.ckpt

    INPUT_DIR = args.target
    OUTPUT_DIR = args.output

    requantize = args.rq

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    model_sr = codec.sample_rate

    with open(MODEL_PATH, "rb") as file:
        gen_model = pkl.load(file)
    
    # segment_length_in_samples = int(model_sr*4/(SUBDIV*TEMPO))

    input_waves, file_names = uload.load_dir(INPUT_DIR, model_sr)

    uload.mkdir(OUTPUT_DIR)
    
    with torch.inference_mode():
        for input_wave, file_name in zip(input_waves, file_names):
            input_vecs = codec.encode(input_wave.to(DEVICE).unsqueeze(0))[0].transpose(1,2).reshape(-1,1024).cpu().numpy()
            transformed_vecs = gen_model(input_vecs)
            transformed_latents = torch.tensor(np.transpose(transformed_vecs)).reshape(1, 1024, -1).to(DEVICE)

            if requantize:
                transformed_latents = codec.quantizer(transformed_latents)[0]

            output_wave = codec.decode(transformed_latents).reshape(1,-1)

            torchaudio.save(f"{OUTPUT_DIR}/{file_name}", output_wave.detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train GAN-based timbre transfer model using paired query/carget datasets.\n"
    "File pairs must have the same names within their respective directories.\n"
    "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--target", help="Location of input audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--output", help="Output directory.", type=str, metavar="path", required=True)
    parser.add_argument("--ckpt", help="Path to trained affine transformation.", type=str, metavar="path", required=True)
    parser.add_argument("--rq", help="Requantize generated latent sequence", action="store_true")
    args = parser.parse_args()
    main(args)