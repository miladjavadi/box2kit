import numpy as np
import torchaudio
import torch
import dac
import argparse
import os
import pickle
import datetime

from gantransfer.train import load_mono, reshape_dataset
from autoconcat.model import PairedCorpus, AutoConcatenator

def batch_partition(dataset, batch_size: int = 64):
    return [dataset[i:i+batch_size] for i in range(0, dataset.shape[0], batch_size)]

def prepare_corpus(query_dir: str, target_dir: str, block_length_in_samples: int, codec: dac.DAC, device: str, batch_size: int = 64):
    query_files = sorted(os.listdir(query_dir))
    target_files = sorted(os.listdir(target_dir))

    # load in all waveforms
    query_waveforms = [load_mono((f"{query_dir}/{file}"), codec.sample_rate).to(device) for file in query_files if file[-4:] == ".wav"]
    target_waveforms = [load_mono((f"{target_dir}/{file}"), codec.sample_rate).to(device) for file in target_files if file[-4:] == ".wav"]

    query_block_waveforms = reshape_dataset(query_waveforms, block_length_in_samples)
    target_block_waveforms = reshape_dataset(target_waveforms, block_length_in_samples)

    with torch.inference_mode():
        query_dataset = [codec.encode(waveform)["codes"] for waveform in batch_partition(query_block_waveforms, batch_size)] # save latent code indices to save space
        query_dataset = torch.cat(query_dataset, dim=0)

        target_dataset = [codec.encode(waveform)["codes"] for waveform in batch_partition(target_block_waveforms, batch_size)]
        target_dataset = torch.cat(target_dataset, dim=0)

    query_dataset = query_dataset
    target_dataset = target_dataset

    return query_dataset, target_dataset

def main(args):
    query_dir = args.querydir
    target_dir = args.targetdir
    tempo = args.tempo
    subdivs = args.subdiv
    out_dir = args.outdir
    batch_size = args.batchsize

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_filename = f"autoconcat_{timestamp}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model_sr = dac_model.sample_rate
    block_length_in_samples = int(model_sr*60/(tempo*subdivs/4))

    query_blocks, target_blocks = prepare_corpus(query_dir, target_dir, block_length_in_samples, dac_model, device, batch_size)

    corpus = PairedCorpus(query_blocks, target_blocks, block_length_in_samples)
    model = AutoConcatenator(corpus)

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    
    out_path = f"{out_dir}/{out_filename}"

    # with open(out_path, "wb") as file:
    #     pickle.dump(model, file)

    torch.save(model, out_path)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train autoconcatenative timbre transfer model using paired query/carget datasets.\n"
    "File pairs must have the same names within their respective directories.\n"
    "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--querydir", help="Location of query audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--targetdir", help="Location of target audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--batchsize", help="Batch size to encode audio blocks in.", type=int, metavar="blocks", default=64)
    parser.add_argument("--tempo", help="Reference tempo against which to divide audio blocks. Should ideally match the tempo of the audio data.", type=float, metavar="bpm", default=90)
    parser.add_argument("--subdiv", help="Subdivisions against which to divide audio blocks. For instance, \"--tempo 90 --subdiv 8\" means that audio waveforms will be divided into 1/8th note long chunks at 90 BPM.", type=int, metavar="subdivisions", default=8)
    parser.add_argument("--outdir", help="Output directory for saved model.", type=str, metavar="path", default="autoconcat_models")
    args=parser.parse_args()
    main(args)