import dac
import torchaudio
from box2kit.autoconcat.model import PairedCodebook, MatchSearchTransfer
import torch
import argparse
import datetime

import box2kit.utils.load_data as uload

# constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 48000

def main(args):
    BATCH_SIZE = args.batchsize
    TEMPO = args.bpm
    SUBDIVS = args.subdivs
    EXPERIMENT_NAME = args.name
    CODEBOOK_LENGTH = args.len
    SPLIT = args.split

    TARGET_DIR = args.target
    OUTPUT_DIR = args.output

    dac_model = dac.DAC.load(dac.utils.download()).to(DEVICE)
    model_sr = dac_model.sample_rate
    seg_length_in_samples = int(model_sr*60/(TEMPO*SUBDIVS/4))

    target_waves = uload.load_dir(TARGET_DIR, model_sr)
    output_waves = uload.load_dir(OUTPUT_DIR, model_sr)

    target_waveform_segs = uload.reshape_data(target_waves, seg_length_in_samples).to(DEVICE)
    output_waveform_segs = uload.reshape_data(output_waves, seg_length_in_samples).to(DEVICE)

    paired_waveform_segs = torch.cat((target_waveform_segs, output_waveform_segs), dim=1) # cat along channel direction

    train_waveform_data, val_waveform_data = uload.binary_split(paired_waveform_segs, SPLIT)

    train_data = torch.stack([uload.safe_encode(train_waveform_data[:,i,:].unsqueeze(1), dac_model) for i in range(2)], dim=1)
    val_data = torch.stack([uload.safe_encode(val_waveform_data[:,i,:].unsqueeze(1), dac_model) for i in range(2)], dim=1)

    codebook = PairedCodebook(train_data, val_data, seg_length_in_samples, CODEBOOK_LENGTH)

    torch.save(codebook, EXPERIMENT_NAME)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser=argparse.ArgumentParser(description="Generate codebook for match search transfer model.")

    parser.add_argument("--target", help="Location of target instrument training data.", type=str, metavar="path", required=True)
    parser.add_argument("--output", help="Location of output instrument training data.", type=str, metavar="path", required=True)
    parser.add_argument("--batchsize", help="Batch size for point pair distance calculation.", type=int, metavar="size", default=32)
    parser.add_argument("--name", help="Name of codebook.", type=str, metavar="codebook_name", default=f"{uload.mkdir('codebooks')}/{timestamp}")
    parser.add_argument("--bpm", help="Reference tempo against which to divide audio segments. Should ideally match the tempo of the audio data.", type=float, metavar="bpm", default=90)
    parser.add_argument("--subdiv", help="Subdivisions against which to divide audio segments. For instance, \"--tempo 90 --subdiv 8\" means that audio waveforms will be divided into 1/8th note long segments at 90 bpm.", type=int, metavar="subdivisions", default=8)
    parser.add_argument("--len", help="Codebook length.", type=int, metavar="length", default=512)
    parser.add_argument("--split", help="Ratio of training data.", type=float, metavar="ratio", default=0.8)
    args=parser.parse_args()
    main(args)