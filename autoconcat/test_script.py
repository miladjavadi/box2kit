import numpy as np

from model import PairedCorpus, AutoConcatenator

queries = np.arange(-10, 10).reshape(-1, 1, 1) * np.ones((20, 2, 4))

targets = np.arange(40, 60).reshape(-1, 1, 1) * np.ones((20, 2, 4))

corpus = PairedCorpus(queries, targets)
model = AutoConcatenator(corpus)

input = -9.1 * np.ones((2, 4))

output = model.quantize_transfer(input)

print(output)