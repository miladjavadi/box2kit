import torch
import torchaudio
import os

from autoconcat.model import PairedCorpus, AutoConcatenator
from gantransfer.train import load_mono

queries = torch.arange(-10, 10, device="cpu").reshape(-1, 1, 1) * torch.ones((20, 2, 4), device="cpu")

targets = torch.arange(40, 60, device="cpu").reshape(-1, 1, 1) * torch.ones((20, 2, 4), device="cpu")

corpus = PairedCorpus(queries, targets, 2)
model = AutoConcatenator(corpus)

input = 0.4 * torch.ones((2, 4), device="cpu")

output = model.quantize_transfer(input, queries, targets)

print(output)

files = sorted(os.listdir("training_data/beatbox"))

super_vector = torch.zeros((1,0), device="cpu")
for file in files:
    if file[-4:] == ".wav":
        audio = load_mono(f"training_data/beatbox/{file}", 48000).to("cpu")
        super_vector = torch.cat((super_vector, audio), dim=1)

print(super_vector.shape[1]/48000)