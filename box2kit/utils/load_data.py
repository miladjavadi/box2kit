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

    dataset = torch.zeros((0, 1, block_length)).to(waveforms[0].device)
    for waveform in waveforms:
        # trim waveform to whole number of block lengths
        waveform = waveform[:,:((waveform.shape[1]//block_length)*block_length)]

        # reshape waveform into blocks
        blocks = torch.reshape(waveform, (-1, 1, block_length))
        dataset = torch.cat((dataset, blocks), dim=0)
    
    return dataset

def binary_split(data, split=0.8):
    n = data.size(0)
    perm = torch.randperm(n)
    split_n = int(n * split)
    return data[perm[:split_n]], data[perm[split_n:]]

def safe_encode(data, codec, batch_size=8):
    with torch.inference_mode():
        latents = [codec.encode(waveform)[0] for waveform in batch_partition(data, batch_size)]
        latents = torch.cat(latents, dim=0)
    return latents

def safe_decode(data, codec, batch_size=8):
    with torch.inference_mode():
        latents = [codec.decode(latent) for latent in batch_partition(data, batch_size)]
        latents = torch.cat(latents, dim=0)
    return latents

def batch_partition(dataset, batch_size: int = 64):
    return [dataset[i:i+batch_size] for i in range(0, dataset.shape[0], batch_size)]

def mkdir(dir_path: str):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    return dir_path