import dac
import torchaudio
from box2kit.match.model import PairedCodebook, MatchSearchTransfer
import torch
import argparse
import datetime
import os

import box2kit.utils.load_data as uload
from box2kit.utils.load_data import mkdir

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    global_config = configs["global"]
    model_config = configs["match"]

    # Configurables
    TEMPO = model_config["tempo"]
    SUBDIVS = model_config["subdiv"]
    BATCH_SIZE = model_config["batch_size"]
    CODEBOOK_LENGTH = model_config["codebook_length"]

    TRAIN_TARGET_PATH = global_config["training_target_path"]
    TRAIN_OUTPUT_PATH = global_config["training_output_path"]
    VAL_TARGET_PATH = global_config["validation_target_path"]
    VAL_OUTPUT_PATH = global_config["validation_output_path"]

    models_dir = mkdir(global_config["models"])
    logs_dir = mkdir(os.path.join(models_dir, model_config["logs"]))

    # command-line arguments
    DATA_PATH = args.data
    EXPERIMENT_NAME = args.name

    codebook_path = os.path.join(logs_dir, f"{EXPERIMENT_NAME}.pt")

    train_target_dir = os.path.join(DATA_PATH, TRAIN_TARGET_PATH)
    train_output_dir = os.path.join(DATA_PATH, TRAIN_OUTPUT_PATH)
    val_target_dir = os.path.join(DATA_PATH, VAL_OUTPUT_PATH)
    val_output_dir = os.path.join(DATA_PATH, VAL_OUTPUT_PATH)

    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    SAMPLE_RATE = codec.sample_rate
    seg_length_in_samples = int(SAMPLE_RATE*60/(TEMPO*SUBDIVS/4))

    train_target_waves, _ = uload.load_dir(train_target_dir, SAMPLE_RATE)
    train_output_waves, _ = uload.load_dir(train_output_dir, SAMPLE_RATE)
    val_target_waves, _ = uload.load_dir(val_target_dir, SAMPLE_RATE)
    val_output_waves, _ = uload.load_dir(val_output_dir, SAMPLE_RATE)

    train_target_waveform_segs = uload.reshape_data(train_target_waves, seg_length_in_samples).to(DEVICE)
    train_output_waveform_segs = uload.reshape_data(train_output_waves, seg_length_in_samples).to(DEVICE)
    val_target_waveform_segs = uload.reshape_data(val_target_waves, seg_length_in_samples).to(DEVICE)
    val_output_waveform_segs = uload.reshape_data(val_output_waves, seg_length_in_samples).to(DEVICE)

    train_waveform_data = torch.cat((train_target_waveform_segs, train_output_waveform_segs), dim=1) # cat along channel direction
    val_waveform_data = torch.cat((val_target_waveform_segs, val_output_waveform_segs), dim=1)

    train_data = torch.stack([uload.safe_encode(train_waveform_data[:,i,:].unsqueeze(1), codec) for i in range(2)], dim=1)
    val_data = torch.stack([uload.safe_encode(val_waveform_data[:,i,:].unsqueeze(1), codec) for i in range(2)], dim=1)

    codebook = PairedCodebook(train_data, val_data, seg_length_in_samples, CODEBOOK_LENGTH)

    torch.save(codebook, codebook_path)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    parser=argparse.ArgumentParser(description="Generate paired instrument codebook for match search transfer model.")
    parser.add_argument("data", help="Location of training and validaiton data.", type=str, metavar="path")
    parser.add_argument("--name", help="Name of codebook.", type=str, metavar="name", default=timestamp)
    parser.add_argument("--override", help="Temporarily override config using values from a separate override file.", type=str, metavar="override_file_path", default=None)
    
    args=parser.parse_args()

    configs = uload.load_configs("box2kit/configs", args.override)
    main(args, configs)
