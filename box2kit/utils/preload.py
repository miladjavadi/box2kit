from box2kit.utils.load_data import PairedWaveformDataset, safe_encode
import dac

class PairedLatentSequenceDataset(PairedWaveformDataset):
    """
    Extend `PairedWaveformDataset` dataset to contain pre-computed latent sequences for each data point.

    Used in neural model training for increased performance.

    Args:
        target_dir (str): Path to folder of target audio files.
        output_dir (str): Path to folder of output audio files.
        segment_length (int): Desired length of waveform segments.
        sr (int): Sample rate to resample audio data to.
        codec (DAC): Codec used to encode audio into latent data.
        batch_size (int): How many waveform segments to encode at once using `codec`.
    """
    def __init__(self, target_dir: str, output_dir: str, segment_length: int, sr: int = 44100, codec: dac.DAC = dac.DAC.load(dac.utils.download()), batch_size=1):
        super().__init__(target_dir, output_dir, segment_length, sr)

        self.target_latents = safe_encode(self.target_data.to(codec.device), codec, batch_size)
        self.output_latents = safe_encode(self.output_data.to(codec.device), codec, batch_size)
    
    def __len__(self):
        return self.target_data.shape[0]
    
    def __getitem__(self, idx: int):
        x = self.target_data[idx]
        y = self.output_data[idx]
        p = self.target_latents[idx]
        q = self.output_latents[idx]
        return x, y, p, q