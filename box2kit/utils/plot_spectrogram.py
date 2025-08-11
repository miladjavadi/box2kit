import torch
import torchaudio
import box2kit.utils.load_data as uload
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 48000
NFFT = 1024

def plot_spectrogram(signal, xlim: list[float], ylim: list[float], vlim: list[float]) -> None:
    hop_length = NFFT//4
    vmin, vmax = tuple(vlim)
    
    plt.figure(figsize=(8, 3.2), dpi=100)
    spectrogram = torch.stft(signal.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=signal.device)).abs().cpu().numpy()
    plt.pcolormesh(np.arange(spectrogram.shape[-1])*hop_length/SAMPLE_RATE, np.arange(spectrogram.shape[-2])*SAMPLE_RATE/NFFT, spectrogram[0], vmin=vmin, vmax=vmax, cmap="Greys")
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    return

if __name__ == "__main__":
    bb_wave = uload.load_mono("training_data/beatbox/8.wav", SAMPLE_RATE)
    plot_spectrogram(bb_wave, xlim=[0, 3], ylim=[0, 12000], vlim=[0.5, 1.5])

    dk_wave = uload.load_mono("training_data/drum_kit/8.wav", SAMPLE_RATE)
    plot_spectrogram(dk_wave, xlim=[0, 3], ylim=[0, 12000], vlim=[0.5, 1.5])

    plt.show()