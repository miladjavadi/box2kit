import torch
import torchaudio
import box2kit.utils.load_data as uload
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 33000
NFFT = 128

def plot_spectrogram(signal, xlim: list[float], ylim: list[float], vlim: list[float], name: str = None):
    hop_length = NFFT//4
    vmin, vmax = tuple(vlim)
    
    ### wave and spectrogram subplots
    # fig, axs = plt.subplots(2, 1, figsize=(3, 2.4), dpi=120, sharex=True, constrained_layout=True)

    # axs[0].plot(np.arange(signal.shape[-1])/SAMPLE_RATE, signal[0].cpu().numpy(), "k", linewidth=0.6)
    # axs[0].set_ylim(-1.1, 1.1)
    # axs[0].set_ylabel(name)

    # spectrogram = torch.stft(signal.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=signal.device)).abs().cpu().numpy()
    # c = axs[1].pcolormesh(np.arange(spectrogram.shape[-1])*hop_length/SAMPLE_RATE, np.arange(spectrogram.shape[-2])*SAMPLE_RATE/NFFT, spectrogram[0], vmin=vmin, vmax=vmax, cmap="Greys", rasterized=True)
    
    # axs[1].set_xlim(*xlim)
    # axs[1].set_ylim(*ylim)

    # axs[1].set_xlabel(r"$t$")
    # axs[1].set_ylabel(r"$f$")
    # axs[1].set_yticks([0, 4000, 8000, 12000])

    ### spectrogram only (no axes)
    fig, axs = plt.subplots(constrained_layout=True)

    spectrogram = torch.stft(signal.squeeze(1), NFFT, return_complex=True, window=torch.hann_window(NFFT, device=signal.device)).abs().cpu().numpy()
    c = axs.pcolormesh(np.arange(spectrogram.shape[-1])*hop_length/SAMPLE_RATE, np.arange(spectrogram.shape[-2])*SAMPLE_RATE/NFFT, spectrogram[0], vmin=vmin, vmax=vmax, cmap="Greys", rasterized=True)
    
    axs.axis("off")

    axs.set_xlim(*xlim)
    axs.set_ylim(*ylim)

    fig.tight_layout(pad=0)

    return fig, axs, spectrogram

if __name__ == "__main__":
    
    plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    })

    bb_wave = uload.load_mono("training_data/beatbox/8.wav", SAMPLE_RATE)
    fig, axs, spectrogram = plot_spectrogram(bb_wave, xlim=[0, 1.5], ylim=[0, 16000], vlim=[0.05, 0.6], name=r"$x(t)$")
    # np.savetxt("outs/bb_spec.dat", spec, delimiter=" ", fmt='%1.3f')
    fig.savefig("outs/bb_spec.png", bbox_inches="tight", pad_inches=0, transparent=True)
    # plt.imsave("outs/bb_spec.png", spectrogram[0], cmap="Greys", vmin=0.05, vmax=0.6, xlim=[0, 1.5], ylim=[0, 16000])

    dk_wave = uload.load_mono("training_data/drum_kit/8.wav", SAMPLE_RATE)
    fig, axs, spectrogram = plot_spectrogram(dk_wave, xlim=[0, 1.5], ylim=[0, 16000], vlim=[0.05, 0.6], name=r"$y(t)$")
    # np.savetxt("outs/dk_spec.dat", spec, delimiter=" ", fmt='%1.3f')
    fig.savefig("outs/dk_spec.png", bbox_inches="tight", pad_inches=0, transparent=True)
    # plt.imsave("outs/dk_spec.png", spectrogram[0], cmap="Greys", vmin=0.05, vmax=0.6, xlim=[0, 1.5], ylim=[0, 16000])

    plt.show()