import dac
import torchaudio
from autoconcat.model import PairedCodebook, MatchSearchTransfer
import torch

from utils.load_data import load_dir, reshape_data

def train_val_split(data, split=0.8):
    n = data.size(0)
    perm = torch.randperm(n)
    split_n = int(n * split)
    return data[perm[:split_n]], data[perm[split_n:]]

def main():

    tempo = 90
    subdivs = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model_sr = dac_model.sample_rate
    seg_length_in_samples = int(model_sr*60/(tempo*subdivs/4))

    target_waves = load_dir("training_data/beatbox", model_sr)
    output_waves = load_dir("training_data/drum_kit", model_sr)

    target_waveform_segs = reshape_data(target_waves, seg_length_in_samples)
    output_waveform_segs = reshape_data(target_waves, seg_length_in_samples)

    paired_segs = torch.stack((target_waveform_segs, output_waveform_segs), dim=1)

    train_data, val_data = train_val_split(paired_segs, 0.8)

    print(train_data.shape)

if __name__ == "__main__":
    main()
