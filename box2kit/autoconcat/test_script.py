import torch
import torchaudio
import os
import dac

from box2kit.autoconcat.model import PairedCorpus, AutoConcatenator, PairedCodebook, MatchSearchTransfer
from box2kit.gantransfer.train import load_mono

# targets = torch.arange(-10, 10, device="cpu").reshape(-1, 1, 1) * torch.ones((20, 2, 4), device="cpu")

# outputs = torch.arange(40, 60, device="cpu").reshape(-1, 1, 1) * torch.ones((20, 2, 4), device="cpu")

# corpus = PairedCorpus(targets, outputs, 2)
# model = AutoConcatenator(corpus)

# input = 0.4 * torch.ones((2, 4), device="cpu")

# output = model.quantize_transfer(input, targets, outputs)

# print(output)

# files = sorted(os.listdir("training_data/beatbox"))

# super_vector = torch.zeros((1,0), device="cpu")
# for file in files:
#     if file[-4:] == ".wav":
#         audio = load_mono(f"training_data/beatbox/{file}", 48000).to("cpu")
#         super_vector = torch.cat((super_vector, audio), dim=1)

# print(super_vector.shape[1]/48000)

train_targets = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]]).reshape(-1, 1, 2)
train_outputs = torch.tensor([[2, 2], [-2, 2], [-2, -2], [2, -2]]).reshape(-1, 1, 2)
training_set = torch.stack((train_targets, train_outputs), dim=1)

val_targets = torch.tensor([[1, 0.6], [1, -0.6]]).reshape(-1, 1, 2)
val_outputs = torch.tensor([[2, 1], [2, -1]]).reshape(-1, 1, 2)
validation_set = torch.stack((val_targets, val_outputs), dim=1)

# print(training_set, validation_set)

codebook = PairedCodebook(training_set, validation_set, 2)

print("Codebook:", codebook.codebook)

transfer_model = MatchSearchTransfer(codebook)

test_target = torch.tensor([[1222, 43278], [-139, -2948]]).reshape(-1, 1, 2)

print(test_target)

test_output = transfer_model.transfer_sequence(test_target)

print(test_output)


