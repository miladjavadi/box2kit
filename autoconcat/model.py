import numpy as np
import torch
import cvxpy as cp
from autoconcat.util import batch_partition

class PairedCorpus():
    def __init__(self, query_blocks, target_blocks, block_length_in_samples):
        if query_blocks.ndim != 3 or target_blocks.ndim != 3:
            raise ValueError("Dataset arrays must be three-dimensional (B x Nq x T).")
    
        self.query_blocks = query_blocks # B x Nq x T
        self.target_blocks = target_blocks # B x Nq x T
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
    def nbooks(self):
        return self.query_blocks.shape[1]
    
    @property
    def block_length(self):
        return self.query_blocks.shape[2]

        
class AutoConcatenator():
    def __init__(self, corpus: PairedCorpus):
        self.corpus = corpus

    def quantize_transfer(self, input, query_latents, target_latents):

        # differences = query_latents - input
        # distances = torch.linalg.norm(differences, axis=(1, 2))
        # opt_index = torch.argmin(distances)

        # cosine_sims = torch.nn.functional.cosine_similarity(input.unsqueeze(0), query_latents, dim=1)
        # cosine_norms = torch.linalg.norm(cosine_sims, axis=1)

        # opt_index = torch.argmax(cosine_norms/distances)

        xcorr_sims = torch.stack([self.xcorr_similarity(input, query) for query in query_latents])

        opt_index = torch.argmin(xcorr_sims)

        print("xxx")
        print(xcorr_sims[opt_index])
        print(torch.mean(query_latents[opt_index]))

        output = target_latents[opt_index]

        return output
    
    def salt(self, input, query_latents, target_latents, max_steps=1, tolerance=1e-3):

        scaled_query_corpus = 1/max_steps * query_latents
        scaled_target_corpus = 1/max_steps * target_latents

        residual = input
        transformed_block = torch.zeros(input.shape, device=input.device)

        for i in range(max_steps):
            if torch.linalg.norm(residual) < tolerance:
                break

            differences = scaled_query_corpus - residual
            distances = torch.linalg.norm(differences, axis=(1, 2))
            min_index = torch.argmin(distances)

            best_fit_query = scaled_query_corpus[min_index]
            best_fit_target = scaled_target_corpus[min_index]

            transformed_block = transformed_block + best_fit_target
            residual = residual - best_fit_query

        return transformed_block
    
    def xcorr_similarity(self, x, y):
        xcorr = torch.nn.functional.conv1d(x.unsqueeze(0), y.flip(-1).unsqueeze(1), groups=x.shape[0])
        # print(xcorr.shape)
        mean_xcorr = torch.mean(xcorr, dim=2)
        # print(mean_xcorr.shape)
        xcorr_sim = torch.linalg.norm(mean_xcorr, dim=1)
        # print(xcorr_sim.shape)

        return xcorr_sim
    
    def autoconcat(self, input_waveform, codec, batch_size: int = 64, max_steps: int = 1, tolerance: float = 1e-3, noise=0):
        # input query waveform -> reconstructed target waveform
        query_latents = self.query_latents(codec, noise)
        target_latents = self.target_latents(codec)

        # trim waveform to whole number of block lengths
        input_waveform = input_waveform[:,:((input_waveform.shape[1]//self.corpus.block_length_in_samples)*self.corpus.block_length_in_samples)]

        # reshape waveform into blocks
        input_blocks = torch.reshape(input_waveform, (-1, self.corpus.block_length_in_samples))
        input_blocks = input_blocks.unsqueeze(1)

        with torch.inference_mode():
            batched_input_blocks = batch_partition(input_blocks, batch_size)
            batched_input_embeddings = [codec.encode(batch)[0] for batch in batched_input_blocks]
            input_embeddings = torch.cat(batched_input_embeddings, dim=0)

            transformed_embeddings = torch.empty((0, input_embeddings.shape[1], self.block_length), device=input_waveform.device)

            for block in input_embeddings:
                # transformed_embeddings = torch.cat((transformed_embeddings, self.salt(block, query_latents, target_latents, max_steps, tolerance).unsqueeze(0)), dim=0)
                # transformed_embeddings = torch.cat((transformed_embeddings, self.quantize_transfer(block, query_latents, target_latents).unsqueeze(0)), dim=0)
                transformed_embeddings = torch.cat((transformed_embeddings, self.quantize_transfer(block, query_latents, target_latents).unsqueeze(0)), dim=0)

            batched_transformed_embeddings = batch_partition(transformed_embeddings, batch_size)
            batched_reconstruction = [codec.decode(batch) for batch in batched_transformed_embeddings]


        reconstructed_waveform = torch.cat(batched_reconstruction, dim=0).flatten()

        return reconstructed_waveform
    
    def query_latents(self, codec, noise = 0):
        latents = self.code_to_latents(self.corpus.query_blocks, codec)
        latents = latents + noise*torch.randn_like(latents)
        return latents
    
    def target_latents(self, codec):
        latents = self.code_to_latents(self.corpus.target_blocks, codec)
        return latents
    
    def code_to_latents(self, codes, codec, batch_size: int = 64):
        code_batches = batch_partition(codes, batch_size)
        with torch.inference_mode():
            batched_latents = [codec.quantizer.from_codes(code_batch)[0] for code_batch in code_batches]
        latents = torch.cat(batched_latents, dim=0)
        return latents

    @property
    def nblocks(self):
        return self.corpus.nblocks

    @property
    def nbooks(self):
        return self.corpus.nbooks
    
    @property
    def block_length(self):
        return self.corpus.block_length