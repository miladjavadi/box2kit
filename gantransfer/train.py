import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import datetime

import dac
import torchaudio
from ganmodel import Generator, Discriminator, PairedWaveformDataset

# Load audio file
def load_mono(file_name: str, target_sr: int) -> torch.FloatTensor:
    audio, sr = torchaudio.load(file_name)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    audio = audio.clamp(-1, 1)
    return audio

def reshape_dataset(waveforms: list[torch.FloatTensor], block_length_in_samples: int) -> torch.FloatTensor:
    # List([1 x waveform_length]) -> [n_blocks x 1 x block_lengths_in_samples]

    dataset = torch.zeros((0, block_length_in_samples), device=waveforms[0].device)
    for waveform in waveforms:
        # trim waveform to whole number of block lengths
        waveform = waveform[:,:((waveform.shape[1]//block_length_in_samples)*block_length_in_samples)]

        # reshape waveform into blocks
        blocks = torch.reshape(waveform, (-1, block_length_in_samples))
        dataset = torch.cat((dataset, blocks), dim=0)
    
    dataset = dataset.unsqueeze(1)
    return dataset

def prepare_dataloader(query_dir: str, target_dir: str, block_length_in_samples: int, batch_size: int, model_sr: int, device: str):
    query_files = os.listdir(query_dir)
    target_files = os.listdir(target_dir)

    # load in all waveforms
    query_waveforms = [load_mono((f"{query_dir}/{file}"), model_sr).to(device) for file in query_files if file[-4:] == ".wav"]
    target_waveforms = [load_mono((f"{target_dir}/{file}"), model_sr).to(device) for file in target_files if file[-4:] == ".wav"]

    query_dataset = reshape_dataset(query_waveforms, block_length_in_samples)
    target_dataset = reshape_dataset(target_waveforms, block_length_in_samples)
    paired_dataset = PairedWaveformDataset(query_dataset, target_dataset)
    dataloader = torch.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def training_procedure(gen_model, discr_model, dac_model, dataloader, epochs, device):
    embedding_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCELoss()
    lambda_embedding = 100

    gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.00002, betas=(0.5, 0.999))
    discr_optimizer = optim.Adam(discr_model.parameters(), lr=0.00002, betas=(0.5, 0.999))

    real_label = 1
    fake_label = 0

    for i in range(epochs):
        for batch_nr, (query, target) in enumerate(dataloader):
            print(f"Epoch: {i+1}/{epochs}, Batch: {batch_nr+1}/{len(dataloader)}")

            with torch.no_grad():
                Z_query = dac_model.encode(query)[0]
                Z_target = dac_model.encode(target)[0]

            # train discriminator
            Z_transformed = gen_model(Z_query).detach()
            
            with torch.no_grad():
                transformed_decoded = dac_model.decode(Z_transformed)

            target = target[:,:,:transformed_decoded.shape[2]] # trim tail of target that is lost when decoding
            
            d_real = discr_model(Z_query, target)
            d_fake = discr_model(Z_query, transformed_decoded)

            real_labels = torch.full(d_real.shape, real_label, device=device, dtype=torch.float32)
            real_adversarial_loss = adversarial_loss_fn(d_real, real_labels)

            fake_labels = torch.full(d_fake.shape, fake_label, device=device, dtype=torch.float32)
            fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

            discr_loss = real_adversarial_loss + fake_adversarial_loss
            discr_optimizer.zero_grad()
            discr_loss.backward(retain_graph=True)
            discr_optimizer.step()

            # train generator
            
            Z_transformed = gen_model(Z_query)
            embedding_loss = embedding_loss_fn(Z_transformed, Z_target)

            d_fake = discr_model(Z_query, transformed_decoded)
            fake_adversarial_loss = adversarial_loss_fn(d_fake, fake_labels)

            gen_loss = 1/fake_adversarial_loss + lambda_embedding * embedding_loss
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

def main(args):
    query_dir = args.querydir
    target_dir = args.targetdir
    tempo = args.tempo
    subdivs = args.subdiv
    batch_size = args.batchsize
    max_epochs = args.maxepochs
    chkpt_dir = args.chkptdir

    timecode = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dac_model = dac.DAC.load(dac.utils.download()).to(device)
    model_sr = dac_model.sample_rate
    block_length_in_samples = int(model_sr*60/(tempo*subdivs/4))

    dataloader = prepare_dataloader(query_dir, target_dir, block_length_in_samples, batch_size, model_sr, device)

    # the length of an audio block may be altered during decoding.
    # thus, a second block sample length must be passed to the discriminator
    with torch.inference_mode():
        dummy_frame = dac_model.encode(dataloader.dataset[0][0].unsqueeze(0))[0]
        block_length_in_frames = dummy_frame.shape[2]
        output_block_length_in_samples = dac_model.decode(dummy_frame).shape[2]

    gen_model = Generator().to(device)
    discr_model = Discriminator(output_block_length_in_samples, block_length_in_frames).to(device)

    training_procedure(gen_model, discr_model, dac_model, dataloader, max_epochs, device)

    torch.save({"gen_model": gen_model.state_dict(), "discr_model": discr_model.state_dict(), "block_length_in_samples": output_block_length_in_samples, "block_length_in_frames": block_length_in_frames}, f"{chkpt_dir}/model_{timecode}.pth")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Train GAN-based timbre transfer model using paired query/carget datasets.\n"
    "File pairs must have the same names within their respective directories.\n"
    "For instance: <query_dir>/x.wav should have a corresponding <target_dir>/x.wav.")

    parser.add_argument("--querydir", help="Location of query audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--targetdir", help="Location of target audio files.", type=str, metavar="path", required=True)
    parser.add_argument("--tempo", help="Reference tempo against which to divide audio blocks. Should ideally match the tempo of the audio data.", type=float, metavar="bpm", default=90)
    parser.add_argument("--subdiv", help="Subdivisions against which to divide audio blocks. For instance, \"--tempo 90 --subdiv 8\" means that audio waveforms will be divided into 1/8th note long chunks at 90 BPM.", type=int, metavar="subdivisions", nargs=1, default=8)
    parser.add_argument("--batchsize", help="Number of data point pairs per mini-batch.", type=int, metavar="batch_size", default=16)
    parser.add_argument("--maxepochs", help="Maximum number of training epochs.", type=int, metavar="epochs", default=1000)
    parser.add_argument("--chkptdir", help="Location of save checkpoints.", type=str, metavar="path", default="models")
    args=parser.parse_args()
    main(args)

