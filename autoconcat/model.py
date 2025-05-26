import numpy as np
import torch
import cvxpy as cp

class PairedCorpus():
    def __init__(self, query_blocks, target_blocks, block_length_in_samples):
        if query_blocks.ndim != 3 or target_blocks.ndim != 3:
            raise ValueError("Dataset arrays must be three-dimensional (N x D x T).")
    
        self.query_blocks = query_blocks # N x D x T
        self.target_blocks = target_blocks # N x D x T
        self.block_length_in_samples = block_length_in_samples
    
    def append(self, new_query_blocks, new_target_blocks) -> None:
        if new_query_blocks.shape != new_target_blocks:
            raise ValueError(f"Paired query block and target block dimensions do not match ({new_query_blocks.shape} and {new_target_blocks.shape}).")
        
        try:
            self.query_blocks = torch.cat((self.query_blocks, new_query_blocks), axis=0)
        except ValueError:
            raise ValueError(f"New query blocks must match dimensionality with existing corpus in all dimension except 0 ({self.query_blocks.shape} and {new_query_blocks.shape}).")
        
        try:
            self.target_blocks = torch.cat((self.target_blocks, new_target_blocks), axis=0)
        except ValueError:
            raise ValueError(f"New target blocks must match dimensionality with existing corpus in all dimension except 0 ({self.target_blocks.shape} and {new_target_blocks.shape}).")
    
    @property
    def nblocks(self):
        return self.query_blocks.shape[0]
    
    @property
    def ndims(self):
        return self.query_blocks.shape[1]
    
    @property
    def block_length(self):
        return self.query_blocks.shape[2]

        
class AutoConcatenator():
    def __init__(self, corpus: PairedCorpus):
        self.corpus = corpus

    def quantize_transfer(self, input):
        differences = self.corpus.query_blocks - input
        distances = torch.linalg.norm(differences, axis=(1, 2))
        min_index = torch.argmin(distances)

        output = self.corpus.target_blocks[min_index]

        return output
    
    def salt(self, input):
        device = input.device

        query_blocks = np.asarray(self.corpus.query_blocks)
        target_blocks = np.asarray(self.corpus.target_blocks)
        input = np.asarray(input)

        alpha = cp.Variable(self.nblocks)

        decomposition = cp.sum([alpha[i] * query_blocks[i] for i in range(self.nblocks)])

        objective = cp.Minimize(cp.norm1(alpha))

        constraints = [cp.norm(decomposition - input, 'fro') <= 1e-5]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {prob.status}")

        coeffs = alpha.value
        transformed_block = sum(coeffs[i] * target_blocks[i] for i in range(self.nblocks))

        transformed_block = torch.from_numpy(transformed_block).to(device)

        return transformed_block
    
    @property
    def nblocks(self):
        return self.corpus.nblocks

    @property
    def ndims(self):
        return self.corpus.ndims
    
    @property
    def block_length(self):
        return self.corpus.block_length