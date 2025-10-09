from box2kit.utils.load_data import PairedWaveformDataset, safe_encode
import dac


class PairedLatentSequenceDataset(PairedWaveformDataset):
    """
    Extend waveform dataset to contain pre-computed latent sequences for each data point,
    increasing training performance.
    """
    def __init__(self, target_dir, output_dir, segment_length, sr=44100, codec=dac.DAC.load(dac.utils.download()), batch_size=1):
        super().__init__(target_dir, output_dir, segment_length, sr)

        self.target_latents = safe_encode(self.target_data)
        self.output_latents = safe_encode(self.output_data)
    
    def __len__(self):
        return self.target_data.shape[0]
    
    def __getitem__(self, idx: int):
        x = self.target_data[idx]
        y = self.output_data[idx]
        p = self.target_latents[idx]
        q = self.output_latents[idx]
        return x, y, p, q