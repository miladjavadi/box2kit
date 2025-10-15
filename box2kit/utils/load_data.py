import torch
import torchaudio
from torch.utils.data import DataLoader
import os
import yaml

class PairedWaveformDataset(torch.utils.data.Dataset):
    def __init__(self, target_dir, output_dir, segment_length, sr=44100):
        self.target_data = reshape_data(load_dir(target_dir, sr)[0], segment_length)
        self.output_data = reshape_data(load_dir(output_dir, sr)[0], segment_length)

        if self.target_data.shape != self.output_data.shape:
            raise Exception(f"Query dataset and target dataset must have the same size (query dataset has shape {self.target_data.shape}, while target dataset has shape {self.output_data.shape})")
    
    def __len__(self):
        return self.target_data.shape[0]
    
    def __getitem__(self, idx: int):
        x = self.target_data[idx]
        y = self.output_data[idx]
        return x, y


def load_mono(file_name: str, target_sr: int) -> torch.Tensor:
    audio, sr = torchaudio.load(file_name)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    audio = audio.clamp(-1, 1)
    return audio


def load_dir(dir: str, target_sr: int) -> tuple[list[torch.Tensor], list[str]]:
    files = sorted(os.listdir(dir))
    waves = [load_mono((f"{dir}/{file}"), target_sr) for file in files if file[-4:] == ".wav"]
    return waves, files


def reshape_data(waveforms: list[torch.Tensor], block_length: int) -> torch.FloatTensor:
    """
    reshapes list of mono waveforms to a stack of audio segments

    """
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
    """
    encodes audio segments in mini-batches to reduce gpu memory usage

    """
    with torch.inference_mode():
        latents = [codec.encode(waveform)[0] for waveform in batch_partition(data, batch_size)]
        latents = torch.cat(latents, dim=0)
    return latents


def safe_decode(data, codec, batch_size=8):
    """
    decodes laten sequences in mini-batches to reduce gpu memory usage

    """
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


def load_configs(config_dir: str, override_file: str = None) -> dict:
    default_folder = "default"

    with open(os.path.join(config_dir, default_folder, "global.yaml"), "r") as f:
        user_folder = yaml.safe_load(f)["user_config"]
    
    config_folder = os.path.join(config_dir, user_folder) if os.path.exists(os.path.join(config_dir, user_folder)) else os.path.join(config_dir, default_folder)

    file_list = [file for file in os.listdir(config_folder) if file.endswith(".yaml") and not file.startswith(".")]
    configs = {file_name[:-5]: yaml.safe_load(open(os.path.join(config_folder, file_name), "r")) for file_name in file_list}

    if override_file is not None:
        with open(override_file, "r") as f:
            overrides = yaml.safe_load(f)
            for attribute in overrides:
                keys = attribute.split("/")
                old_val = configs[keys[0]][keys[1]]
                new_val = overrides[attribute]

                print(f"Overriding {attribute}: {old_val} -> {new_val}")
                configs[keys[0]][keys[1]] = new_val

    return configs