import torch
import dac
import argparse
import pickle as pkl
import numpy as np

from box2kit.utils import load_data as uload
from box2kit.affine.model import AffineTransfer
import datetime
import os

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args, configs):
    global_config = configs["global"]
    model_config = configs["affine"]

    # Configurables
    TEMPO = args.tempo
    SUBDIV = args.subdiv
    BATCH_SIZE = args.batchsize
    EXP_NAME = args.name

    N_TRIALS = args.trials
    N_SAMPLES = args.samples
    INL_THRESHOLD = args.threshold

    TRAIN_TARGET_PATH = global_config["training_target_path"]
    TRAIN_OUTPUT_PATH = global_config["training_output_path"]
    VAL_TARGET_PATH = global_config["validation_target_path"]
    VAL_OUTPUT_PATH = global_config["validation_output_path"]

    MODELS_PATH = global_config["models"]

    # command-line arguments
    DATA_PATH = args.data
    EXPERIMENT_NAME = args.name

    train_target_dir = os.path.join(DATA_PATH, TRAIN_TARGET_PATH)
    train_output_dir = os.path.join(DATA_PATH, TRAIN_OUTPUT_PATH)
    val_target_dir = os.path.join(DATA_PATH, VAL_OUTPUT_PATH)
    val_output_dir = os.path.join(DATA_PATH, VAL_OUTPUT_PATH)

    save_path = os.path.join(MODELS_PATH, f"{EXPERIMENT_NAME}.pt")

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    SAMPLE_RATE = codec.sample_rate

    segment_length_in_samples = int(SAMPLE_RATE*4*60/(SUBDIV*TEMPO))

    with torch.inference_mode():
        target_waves, _ = uload.load_dir(train_target_dir, SAMPLE_RATE)
        target_waves.extend(uload.load_dir(val_target_dir, SAMPLE_RATE)[0])
        target_segs = uload.reshape_data(target_waves, segment_length_in_samples).to(DEVICE)
        target_latent_segs = uload.safe_encode(target_segs, codec, BATCH_SIZE)
        target_vecs = target_latent_segs.transpose(1,2).reshape(-1,1024).cpu().numpy()

        output_waves, _ = uload.load_dir(train_output_dir, SAMPLE_RATE)
        output_waves.extend(uload.load_dir(val_output_dir, SAMPLE_RATE)[0])
        output_segs = uload.reshape_data(output_waves, segment_length_in_samples).to(DEVICE)
        output_latent_segs = uload.safe_encode(output_segs, codec, BATCH_SIZE)
        output_vecs = output_latent_segs.transpose(1,2).reshape(-1,1024).cpu().numpy()

        gen_model = AffineTransfer(1024)
        gen_model.fit(target_vecs, output_vecs, N_TRIALS, INL_THRESHOLD, N_SAMPLES)

    with open(f"{save_path}.pkl", 'wb') as file:
        pkl.dump(gen_model, file)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    configs = uload.load_configs("box2kit/configs")
    
    parser=argparse.ArgumentParser(description="Generate paired instrument codebook for match search transfer model.")
    parser.add_argument("data", help="Location of training and validaiton data (For this model, training and validation data are combined and used for training).", type=str, metavar="path")
    parser.add_argument("--name", help="Name of codebook.", type=str, metavar="name", default=timestamp)
    
    args=parser.parse_args()
    main(args, configs)