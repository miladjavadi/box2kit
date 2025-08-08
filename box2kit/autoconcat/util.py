import torch

def batch_partition(dataset, batch_size: int = 64):
    return [dataset[i:i+batch_size] for i in range(0, dataset.shape[0], batch_size)]