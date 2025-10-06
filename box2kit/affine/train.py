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
    tempo = args.tempo
    subdiv = args.subdiv
    batch_size = args.batchsize
    experiment_name = args.name

    n_trials = args.trials
    n_samples = args.samples
    inlier_threshold = args.threshold

    train_target_path = global_config["training_target_path"]
    train_output_path = global_config["training_output_path"]
    val_target_path = global_config["validation_target_path"]
    val_output_path = global_config["validation_output_path"]

    models_path = global_config["models"]

    # command-line arguments
    data_path = args.data
    experiment_name = args.name

    train_target_dir = os.path.join(data_path, train_target_path)
    train_output_dir = os.path.join(data_path, train_output_path)
    val_target_dir = os.path.join(data_path, val_output_path)
    val_output_dir = os.path.join(data_path, val_output_path)

    save_path = os.path.join(models_path, f"{experiment_name}.pt")

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    sample_rate = codec.sample_rate

    segment_length_in_samples = int(sample_rate*4*60/(subdiv*tempo))

    with torch.inference_mode():
        target_waves, _ = uload.load_dir(train_target_dir, sample_rate)
        target_waves.extend(uload.load_dir(val_target_dir, sample_rate)[0])
        target_segs = uload.reshape_data(target_waves, segment_length_in_samples).to(DEVICE)
        target_latent_segs = uload.safe_encode(target_segs, codec, batch_size)
        target_vecs = target_latent_segs.transpose(1,2).reshape(-1,1024).cpu().numpy()

        output_waves, _ = uload.load_dir(train_output_dir, sample_rate)
        output_waves.extend(uload.load_dir(val_output_dir, sample_rate)[0])
        output_segs = uload.reshape_data(output_waves, segment_length_in_samples).to(DEVICE)
        output_latent_segs = uload.safe_encode(output_segs, codec, batch_size)
        output_vecs = output_latent_segs.transpose(1,2).reshape(-1,1024).cpu().numpy()

        gen_model = AffineTransfer(1024)
        gen_model.fit(target_vecs, output_vecs, n_trials, inlier_threshold, n_samples)

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