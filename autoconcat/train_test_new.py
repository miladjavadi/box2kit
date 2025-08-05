import dac
import torchaudio
from autoconcat.model import PairedCodebook, MatchSearchTransfer
import torch
from autoconcat.util import batch_partition

from utils.load_data import load_dir, reshape_data

def train_val_split(data, split=0.8):
    n = data.size(0)
    perm = torch.randperm(n)
    split_n = int(n * split)
    return data[perm[:split_n]], data[perm[split_n:]]

def safe_encode(data, codec, batch_size=8):
    with torch.inference_mode():
        latents = [codec.encode(waveform)[0] for waveform in batch_partition(data, batch_size)]
        latents = torch.cat(latents, dim=0)
    return latents

def main():

    tempo = 90
    subdivs = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model_sr = dac_model.sample_rate
    seg_length_in_samples = int(model_sr*60/(tempo*subdivs/4))

    target_waves = load_dir("training_data/beatbox", model_sr)
    output_waves = load_dir("training_data/drum_kit", model_sr)

    target_waveform_segs = reshape_data(target_waves, seg_length_in_samples).to(device)
    output_waveform_segs = reshape_data(target_waves, seg_length_in_samples).to(device)

    paired_waveform_segs = torch.cat((target_waveform_segs, output_waveform_segs), dim=1) # cat along channel direction

    train_waveform_data, val_waveform_data = train_val_split(paired_waveform_segs, 0.8)

    train_data = torch.stack([safe_encode(train_waveform_data[:,i,:].unsqueeze(1), dac_model) for i in range(2)], dim=1)
    val_data = torch.stack([safe_encode(val_waveform_data[:,i,:].unsqueeze(1), dac_model) for i in range(2)], dim=1)

    codebook = PairedCodebook(train_data, val_data, 512)

    print(codebook.codebook.shape)
    gen_model = MatchSearchTransfer(codebook)

    

if __name__ == "__main__":
    main()
