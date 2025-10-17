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
    """
    Load audio file as Torch tensor, averaged to mono.

    The audio is resampled and clipped at 0dB.

    Args:
        file_name (str): Name of audio file to load
        target_sr (int): Sample rate to resample audio to.

    Returns:
        audio (Tensor): Loaded mono waveform.
    """
    audio, sr = torchaudio.load(file_name)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    audio = audio.clamp(-1, 1)
    return audio


def load_dir(dir: str, target_sr: int) -> tuple[list[torch.Tensor], list[str]]:
    """
    Load folder of `.wav` files as list of Torch tensors.

    Each audio file is averaged to mono, resampled and clipped at 0dB.

    .. seealso::
        `load_mono` loads a single audio file as a tensor.

    Args:
        dir (str): Path to folder of `.wav` files to load.
        target_sr (int): Sample rate to resample audio to.
    
    Returns:
        waves (list of Tensor): Loaded waveforms.
        wave_files (list of str): File names of loaded audio.
    """
    files = sorted(os.listdir(dir))
    wav_files = [file for file in files if file[-4:] == ".wav"]
    waves = [load_mono((f"{dir}/{file}"), target_sr) for file in wav_files]
    return waves, wav_files


def reshape_data(waveforms: list[torch.Tensor], segment_length: int) -> torch.FloatTensor:
    """
    Reshape list of mono waveforms to a stack of equal-sized waveform segments.

    .. note::
        Each waveform is trimmed from the end at dim -1 to the nearest integer multiple of `block_length`. 

    .. examples::
        Basic usage:

        >>> x = [torch.ones(1, 256) for i in range(4)]
    [tensor([[1, ..., 1]]), tensor([[1, ..., 1]]), tensor([[1, ..., 1]]), tensor([[1, ..., 1]])]
    >>> y = reshape_data(x, 128)
    tensor([[[1, ..., 1]], ..., [[1, ..., 1]]])
    >>> y.shape
    tensor([8, 1, 128])

    Args:
        waveforms (list of Tensor): List of waveforms to reshape.
        segment_length (int): Length of each waveform segment.
    """
    # List([1 x waveform_length]) -> [n_blocks x 1 x block_lengths]

    dataset = torch.zeros((0, 1, segment_length)).to(waveforms[0].device)
    for waveform in waveforms:
        # trim waveform to whole number of block lengths
        waveform = waveform[:,:((waveform.shape[1]//segment_length)*segment_length)]

        # reshape waveform into blocks
        blocks = torch.reshape(waveform, (-1, 1, segment_length))
        dataset = torch.cat((dataset, blocks), dim=0)
    
    return dataset


def binary_split(data: torch.Tensor, split: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly split array along dim 0 into two sub-arrays.

    `split` determines the relative size of the first return sub-array.

    Args:
        data (Tensor): Array to split.
        split (float): Relative split between sub-arrays
    
    Returns:
        Resulting sub-arrays from split.
    """
    n = data.size(0)
    perm = torch.randperm(n)
    split_n = int(n * split)
    return data[perm[:split_n]], data[perm[split_n:]]


def safe_encode(data, codec, batch_size=8) -> torch.Tensor:
    """
    encodes audio segments in mini-batches to reduce gpu memory usage

    """
    with torch.inference_mode():
        latents = [codec.encode(waveform)[0] for waveform in batch_partition(data, batch_size)]
        latents = torch.cat(latents, dim=0)
    return latents


def safe_decode(data, codec, batch_size=8) -> torch.Tensor:
    """
    decodes laten sequences in mini-batches to reduce gpu memory usage

    """
    with torch.inference_mode():
        latents = [codec.decode(latent) for latent in batch_partition(data, batch_size)]
        latents = torch.cat(latents, dim=0)
    return latents


def batch_partition(dataset: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """
    Partition stacked data points into list of mini-batches.

    .. examples::
        Basic usage:
        
        >>> x = torch.ones(4, 1, 2)
    tensor([[[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]])
    >>> y = batch_partition(x, 1)
    [tensor([[[1], [1]]]), tensor([[[1], [1]]]), tensor([[[1], [1]]]), tensor([[[1], [1]]])]

    Args:
        dataset (Tensor): Data point stack to partition.
        batch_size (int): Number of data points in each mini batch
    
    Returns:
        List of mini-batches.
    """
    return [dataset[i:i+batch_size] for i in range(0, dataset.shape[0], batch_size)]


def mkdir(dir_path: str) -> str:
    """
    Create new directory if it does not already exist.

    Args:
        dir_path (str): Path to new directory.
    
    Returns:
        dir_path (str): Path to new directory.
    """
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    return dir_path


def load_configs(config_dir: str, override_file: str = None) -> dict:
    """
    Load folder configuration files as dict.

    The loaded dict will consist of multiple dict-valued entries,
    whose keys correspond to the name of each configuration file,
    and contain the file's respective configuration parameters.

    Args:
        config_dir (str): Path to folder containing `.yaml` configuration files.

    Returns:
        configs (dict): Loaded configuration parameters.
    """
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

def match_trim(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trim two tensor at the end of dim -1 such that their dimensions match that of the shorter tensor.

    .. examples::
        Basic usage:

        >>> x = torch.arange(4)
        tensor([0, 1, 2, 3])
        >>> y = torch.arange(5)
        tensor([0, 1, 2, 3, 4])
        >>> match_trim(x, y)
        (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))

    Args:
        x (Tensor): First input tensor.
        y (Tensor): Second input tensor.
    
    Returns:
        Trimmed tensors.
    """
    if x.shape[-1] > y.shape[-1]:
        x = x[..., :y.shape[-1]]
    elif x.shape[-1] < y.shape[-1]:
        y = y[..., :x.shape[-1]]
    
    return x, y