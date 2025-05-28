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
    salt_max_steps = args.steps
    salt_tolerance = 10**(-args.tolerance)
    batch_size = args.batchsize
    noise = args.noise # DEBUG

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model_sr = dac_model.sample_rate

    input_waveform = load_mono(input_path, model_sr).to(device)

    transfer_output = ckpt.autoconcat(input_waveform, dac_model, batch_size, salt_max_steps, salt_tolerance, noise)

    torchaudio.save(output_file_name, transfer_output.unsqueeze(0).detach().cpu(), model_sr)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained autoconcatonator model.")
    # parser=argparse.ArgumentParser(description="Transform input audio data using pre-trained generator model.\n"
    # "File pairs must have the same names within their respective directories.\n"
    # "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--input", help="Path to input audio file.", type=str, metavar="path", required=True)
    parser.add_argument("--ckpt", help="Path to checkpoint folder.", type=str, metavar="path", required=True)
    parser.add_argument("--out", help="Name of output file.", type=str, metavar="name", required=True)
    parser.add_argument("--batchsize", help="Batch size to encode audio blocks in.", type=int, metavar="blocks", default=64)
    parser.add_argument("--salt", help="Use SpArse Linear Transformation (SALT) instead of corpus quantization.", action="store_true")
    parser.add_argument("--tolerance", help="SALT greedy search early-stopping tolerance (in negative powers of 10).", type=float, metavar="pow", default=3)
    parser.add_argument("--steps", help="Max steps for SALT greedy search.", type=int, metavar="steps", default=4)
    parser.add_argument("--noise", help="(DEBUG) How much noise to add to the corpus query blocks", type=float, metavar="pow", default=0)
    args=parser.parse_args()
    main(args)