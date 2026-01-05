import torch
import numpy as np
import matplotlib.pyplot as plt
from box2kit.utils.load_data import load_mono

SAMPLE_RATE = 48000

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

def rms_am(signal, window_size=120):

    num_windows = len(signal) // window_size

    t = np.arange(num_windows*4) / (SAMPLE_RATE*4/(window_size))

    windows = signal[:num_windows * window_size].reshape(num_windows, window_size)
    rms_values = np.sqrt(np.mean(windows**2, axis=1))

    # normalize
    rms_values = rms_values/np.max(rms_values)

    # clip
    rms_values = np.clip(rms_values, 0, 0.25)

    # normalize
    rms_values = rms_values/np.max(rms_values)

    modulator = np.repeat(rms_values, 4)

    carrier = np.sin((SAMPLE_RATE*4/(window_size))*t)

    output = carrier * modulator
    
    return t, output
    

if __name__=="__main__":
    start_time = 8
    end_time = 10
    start_sample = int(SAMPLE_RATE*start_time)
    end_sample = int(SAMPLE_RATE*end_time)
    wave = load_mono("outs/testing_outputs2/t8g/16.wav", SAMPLE_RATE).cpu().numpy()[0][start_sample:end_sample]

    # plot_fft(wave[0], SAMPLE_RATE)

    # t = np.arange(len(wave)) / SAMPLE_RATE
    # plt.plot(t, wave)
    # np.savetxt("outs/linn_wave.dat", np.column_stack((t, wave)), fmt="%.6f")

    t, rms_wave = rms_am(wave, 1000)
    t = t + start_time # time offset
    # rms_wave = rms_wave*2.4

    # plt.plot(t, rms_wave)
    # np.savetxt("outs/target_snare.dat", np.column_stack((t, rms_wave)), fmt="%.6f")

    # t = np.linspace(start_time, end_time, end_sample-start_sample)

    plt.plot(t, rms_wave)
    np.savetxt("outs/t8g_intro.dat", np.column_stack((t, rms_wave)), fmt="%.6f")

    plt.show()
