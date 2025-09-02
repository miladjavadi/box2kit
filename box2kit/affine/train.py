import torch
import dac
import argparse
import pickle as pkl

from box2kit.utils import load_data as uload
from box2kit.affine.model import AffineTransfer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    TEMPO = args.tempo
    SUBDIV = args.subdiv
    BATCH_SIZE = args.batchsize
    EXP_NAME = args.name

    N_TRIALS = args.trials
    N_SAMPLES = args.samples
    INL_THRESHOLD = args.threshold

    TARGET_DIR = args.target
    OUTPUT_DIR = args.output

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    model_sr = codec.sample_rate

    segment_length_in_samples = int(model_sr*4/(SUBDIV*TEMPO))

    with torch.inference_mode():
        target_waves, _ = uload.load_dir(TARGET_DIR, model_sr)
        target_segs = uload.reshape_data(target_waves, segment_length_in_samples)
        target_latent_segs = uload.safe_encode(target_segs, codec, BATCH_SIZE)
        target_vecs = target_latent_segs.transpose(1,2).reshape(-1,1024).cpu().numpy()

        output_waves, _ = uload.load_dir(OUTPUT_DIR, model_sr)
        output_segs = uload.reshape_data(output_waves, segment_length_in_samples)
        output_latent_segs = uload.safe_encode(output_segs, codec, BATCH_SIZE)
        output_vecs = output_latent_segs.transpose(1,2).reshape(-1,1024).cpu().numpy()

        gen_model = AffineTransfer(1024)
        gen_model.fit(target_vecs, output_vecs, N_TRIALS, INL_THRESHOLD, N_SAMPLES)

    with open(f"{EXP_NAME}.pkl", 'w') as file:
        pkl.dump(gen_model, file)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train GAN-based timbre transfer model using paired query/carget datasets.\n"
    "File pairs must have the same names within their respective directories.\n"
    "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--target", help="Location of target audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--output", help="Location of output audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--tempo", help="Reference tempo against which to divide audio blocks. Should ideally match the tempo of the audio data.", type=float, metavar="bpm", default=90)
    parser.add_argument("--subdiv", help="Subdivisions against which to divide audio blocks. For instance, \"--tempo 90 --subdiv 8\" means that audio waveforms will be divided into 1/8th note long chunks at 90 BPM.", type=int, metavar="subdivisions", default=8)
    parser.add_argument("--batchsize", help="Number of data point pairs per mini-batch.", type=int, metavar="batch_size", default=16)
    parser.add_argument("--name", help="Name of experiment.", type=str, metavar="experiment_name", default="affine_transform")
    parser.add_argument("--trials", help="Number of RANSAC candidates.", type=int, metavar="candidates", default=100)
    parser.add_argument("--samples", help="Number of points used to derive each candidate transform.", type=int, metavar="points", default=256)
    parser.add_argument("--threshold", help="L2 distance threshold for inliers.", type=int, default=100)
    args = parser.parse_args()
    main(args)