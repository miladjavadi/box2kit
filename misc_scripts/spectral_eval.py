import audiotools
from box2kit.utils.load_data import load_dir
import pandas as pd
from dac.nn.loss import MultiScaleSTFTLoss, MelSpectrogramLoss
from torch import nn
import argparse

SAMPLE_RATE = 44100
def main(args):
    gen_dir = args.gen
    ref_dir = args.ref

    gen_waves, gen_fnames = load_dir(gen_dir, SAMPLE_RATE)
    ref_waves, ref_fnames = load_dir(ref_dir, SAMPLE_RATE)

    stft_loss_fn = MultiScaleSTFTLoss([2048, 1024, 512, 256, 128, 64])
    mel_loss_fn = MelSpectrogramLoss(window_lengths=[32, 64, 128, 256, 512, 1024, 2048], n_mels = [5, 10, 20, 40, 80, 160, 320], mel_fmin=[0], mel_fmax=[None], loss_fn=nn.MSELoss())

    if gen_fnames != ref_fnames:
        raise Exception("File-name of generated audio folder must be identical to that of refernce wave audio folder.")

    dists = []

    for (gen_wave, ref_wave, fname) in zip(gen_waves, ref_waves, gen_fnames):
        gen_AS = audiotools.AudioSignal(gen_wave, SAMPLE_RATE)
        ref_AS = audiotools.AudioSignal(ref_wave, SAMPLE_RATE)

        stft_dist = stft_loss_fn(gen_AS, ref_AS)
        mel_dist = mel_loss_fn(gen_AS, ref_AS)

        dists.append({"fname": fname, "stft": stft_dist, "mel": mel_dist})
    
    dists_df = pd.DataFrame.from_dict(dists)
    print(dists_df)

    print("Mean STFT distance:", dists_df["stft"].mean())
    print("Mean Mel distance:", dists_df["mel"].mean())

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("gen", help="Location of generated audio files.", metavar="gen_folder_path", type=str)
    parser.add_argument("ref", help="Location of reference audio files.", metavar="ref_folder_path", type=str)
    args = parser.parse_args()

    main(args)