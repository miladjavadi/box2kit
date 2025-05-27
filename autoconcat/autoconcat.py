import torch
import torchaudio
import dac
import pickle
import argparse

from autoconcat.model import AutoConcatenator
from gantransfer.train import load_mono

def main(args):
    input_path = args.input
    output_file_name = args.out
    ckpt_path = args.ckpt
    use_salt = args.salt
    
    # with open(ckpt_path, "rb"):
    #     ckpt = pickle.load(ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    print(type(ckpt.corpus.query_blocks))

    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model_sr = dac_model.sample_rate

    block_length_in_samples = ckpt.corpus.block_length_in_samples

    input_waveform = load_mono(input_path, model_sr).to(device)
    # trim waveform to whole number of block lengths
    input_waveform = input_waveform[:,:((input_waveform.shape[1]//block_length_in_samples)*block_length_in_samples)]

    # reshape waveform into blocks
    input_blocks = torch.reshape(input_waveform, (-1, block_length_in_samples))
    input_blocks = input_blocks.unsqueeze(1)

    with torch.inference_mode():
        input_embeddings = dac_model.encode(input_blocks)[0]

        transformed_embeddings = torch.empty((0, ckpt.ndims, ckpt.block_length), device=device)

        if use_salt:
            for block in input_embeddings:
                transformed_embeddings = torch.cat((transformed_embeddings, ckpt.salt(block).unsqueeze(0)), dim=0)
        else:
            for block in input_embeddings:
                transformed_embeddings = torch.cat((transformed_embeddings, ckpt.quantize_transfer(block).unsqueeze(0)), dim=0)

        reconstruction = dac_model.decode(transformed_embeddings)

    reconstructed_waveform = reconstruction.flatten()

    torchaudio.save(output_file_name, reconstructed_waveform.unsqueeze(0).detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained autoconcatonator model.")
    # parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained generator model.\n"
    # "File pairs must have the same names within their respective directories.\n"
    # "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--input", help="Path to input audio file.", type=str, metavar="path", required=True)
    parser.add_argument("--ckpt", help="Path to checkpoint folder.", type=str, metavar="path", required=True)
    parser.add_argument("--out", help="Name of output file.", type=str, metavar="name", required=True)
    parser.add_argument("--salt", help="Use SpArse Linear Transformation (SALT) instead of corpus quantization.", action="store_true")
    args=parser.parse_args()
    main(args)