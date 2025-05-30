import torch
import torchaudio
from torch.utils.data import DataLoader
import os

def load_mono(file_name: str, target_sr: int) -> torch.FloatTensor:
    audio, sr = torchaudio.load(file_name)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    audio = audio.clamp(-1, 1)
    return audio

def load_dir(dir: str, target_sr: int) -> list[torch.FloatTensor]:
    files = sorted(os.listdir(dir))
    waves = [load_mono((f"{dir}/{file}"), target_sr) for file in files if file[-4:] == ".wav"]
    return waves

def reshape_data(waveforms: list[torch.FloatTensor], block_length: int) -> torch.FloatTensor:
    # List([1 x waveform_length]) -> [n_blocks x 1 x block_lengths]

    dataset = torch.zeros((0, 1, block_length))
    for waveform in waveforms:
        # trim waveform to whole number of block lengths
        waveform = waveform[:,:((waveform.shape[1]//block_length)*block_length)]

        # reshape waveform into blocks
        blocks = torch.reshape(waveform, (-1, 1, block_length))
        dataset = torch.cat((dataset, blocks), dim=0)
    
    return dataset