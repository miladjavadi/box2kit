import torch
import numpy as np
import matplotlib.pyplot as plt
from box2kit.utils.load_data import load_mono

def plot_fft(signal, sample_rate):
    # signal: 1D numpy array (e.g., a single audio channel)
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    fft_spectrum = np.abs(np.fft.rfft(signal))

    plt.plot(freqs, fft_spectrum)
    plt.title("FFT of VAE Output")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 500)  # focus on low freqs
    plt.grid()
    # plt.savefig("outs/roots.png")
    plt.show()

SAMPLE_RATE = 48000
if __name__=="__main__":
    wave = load_mono("outs/box2kit_test/epoch_2500.wav", SAMPLE_RATE).cpu().numpy()

    plot_fft(wave[0], SAMPLE_RATE)
