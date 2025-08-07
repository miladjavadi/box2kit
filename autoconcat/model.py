import numpy as np
import torch
from autoconcat.util import batch_partition
from svae.model import PairedWaveformDataset

class PairedCorpus():
    def __init__(self, target_blocks, output_blocks, block_length_in_samples):
        if target_blocks.ndim != 3 or output_blocks.ndim != 3:
            raise ValueError("Dataset arrays must be three-dimensional (B x Nq x T).")
    
        self.target_blocks = target_blocks # B x Nq x T
        self.output_blocks = output_blocks # B x Nq x T
        self.block_length_in_samples = block_length_in_samples
    
    def append(self, new_target_blocks, new_output_blocks) -> None:
        if new_target_blocks.shape != new_output_blocks:
            raise ValueError(f"Paired target block and output block dimensions do not match ({new_target_blocks.shape} and {new_output_blocks.shape}).")
        
        try:
            self.target_blocks = torch.cat((self.target_blocks, new_target_blocks), axis=0)
        except ValueError:
            raise ValueError(f"New target blocks must match dimensionality with existing corpus in all dimension except 0 ({self.target_blocks.shape} and {new_target_blocks.shape}).")
        
        try:
            self.output_blocks = torch.cat((self.output_blocks, new_output_blocks), axis=0)
        except ValueError:
            raise ValueError(f"New output blocks must match dimensionality with existing corpus in all dimension except 0 ({self.output_blocks.shape} and {new_output_blocks.shape}).")
    
    @property
    def nblocks(self):
        return self.target_blocks.shape[0]
    
    @property
    def nbooks(self):
        return self.target_blocks.shape[1]
    
    @property
    def block_length(self):
        return self.target_blocks.shape[2]


class PairedCodebook():
    def __init__(self, training_set: PairedWaveformDataset, validation_set: PairedWaveformDataset, codebook_length: int=512):      
        if codebook_length < len(training_set):
            with torch.no_grad():
                self.codebook = self.greedy_codebook(training_set, validation_set, codebook_length)
        else:
            raise ValueError(f"Desired codebook length exceeds number of training points ({codebook_length} > {len(training_set)}).")
    
    def greedy_codebook(self, training_data, validation_data, codebook_length, batch_size = 64):
        codeword_indices = []

        point_pair_distances = self.point_pair_distance_array(training_data, validation_data, batch_size)
        for i in range(codebook_length):
            print(i)
            new_codeword_index = self.greedy_search_step(codeword_indices, point_pair_distances)
            codeword_indices.append(new_codeword_index)

        codebook = training_data[codeword_indices]
        return codebook
    
    def greedy_search_step(self, codeword_indices, point_pair_distances):
        """
        return codeword index which minimizes match search quantization error between new codebook and validation data.
        tests candidates in batches for increased efficiency :)

        Parameters:
        - codeword_indices: List of indices in current codebook
        - point_pair_distances: Distances between all codeword candidate-validation point pairs

        Outputs:
        - best_index: Index of best performing codeword candidate
        """

        best_score = torch.inf

        codebook_distances = point_pair_distances[codeword_indices]

        codebook_candidate_distances = self.create_codebook_candidates(point_pair_distances, codebook_distances)
        _, best_index = self.batch_greedy_search(codebook_candidate_distances)

        # for candidate in training_data:
        #     candidate_book = torch.cat((codebook, candidate.unsqueeze(0)))
        #     min_distances = torch.tensor([match_search(validation_point[0], candidate_book[:,0])[0] for validation_point in validation_data]).to(training_data.device)
        #     quant_error = torch.mean(min_distances)

        #     if quant_error < best_quant_error:
        #         best_candidate_book = candidate_book
        #         best_quant_error = quant_error

        return best_index
    
    def create_codebook_candidates(self, candidate_distances: torch.Tensor, codebook_distances: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - candidate_distances ([candidate, validation point]): Distances between candidate-validation point pairs.
        - codebook_distances ([codeword, validation point]): Distances between codeword-validation point pairs.

        Outputs:
        - codebook_candidate_distances ([codebook candidate, codeword, validation point]): All codebook candidates' codeword-validation distance pairs.
        """

        # batch_size = candidate_batch_distances.shape[0]
        # expanded_codebook = codebook_distances.unsqueeze(0).expand(batch_size, -1, -1).clone()
        # candidate_codebook_batch = torch.cat((expanded_codebook, candidate_batch_distances.unsqueeze(1)), dim=1)
        # return candidate_codebook_batch

        n_candidates = candidate_distances.shape[0]
        expanded_codebook = codebook_distances.unsqueeze(0).expand(n_candidates, -1, -1)

        codebook_candidate_distances = torch.cat((expanded_codebook, candidate_distances.unsqueeze(1)), dim=1)

        print(codebook_candidate_distances.shape)

        return codebook_candidate_distances

    def batch_greedy_search(self, codebook_candidate_distances: torch.Tensor) -> tuple[float, int]:
        """
        Parameters:
        - codebook_candidate_distances ([codebook candidate, codeword, validation point]): All codebook candidates' codeword-validation distance pairs.

        Outputs:
        - best_score: Mean quantization error of best codebook candidate on validation data
        - best_codebook_index: Batch-relative index of best codebook candidate
        """

        # batch_size = candidate_codebook_batch.shape[0]

        # min_distances = torch.stack([match_search(validation_point[0].unsqueeze(0).expand(batch_size, -1, -1, -1), candidate_codebook_batch[:,:,0])[0] for validation_point in validation_data], dim=1) # [Candidate, Val. Point]
        # quant_errors = torch.mean(min_distances, dim=1) # [Candidate]

        # best_score, best_codebook_index = torch.min(quant_errors, dim=0)

        # return best_score, candidate_codebook_batch[best_codebook_index]

        min_distances, _ = torch.min(codebook_candidate_distances, dim=1)
        scores = torch.mean(min_distances, dim=1)
        best_score, best_codebook_index  = torch.min(scores, dim=0)

        return best_score, best_codebook_index
    
    def point_pair_distance_array(self, training_data, validation_data, batch_size):
        """
        Parameters:
        - training_data ([training pt., domain, dac feat., frame idx.]): Training dataset/codeword candidate set
        - validation_data ([validation pt., domain, dac feat., frame idx.]): Validation set

        Outputs:
        - distances ([training pt., validation pt.]): Distances between training point-validation point pairs
        """
        distances = []

        # calc distances in batches to reduce memory usage
        for i in range(0, training_data.shape[0], batch_size):
            start = i*batch_size
            stop = (i+1)*batch_size
            differences = training_data[start:stop,0,:,:].unsqueeze(1) - validation_data[:,0,:,:].unsqueeze(0)
            distances.append(torch.linalg.norm(differences, axis=(-1, -2)))

        distances = torch.cat(distances)
        return distances
    
    def append(self, new_segment):
        self.codebook = torch.cat(self.codebook, new_segment)
    
    def __len__(self):
        return self.codebook.shape[0]
    
    def __getitem__(self, idx):
        return self.codebook[idx]
    
    @property
    def targets(self):
        return self.codebook[:,0]
    
    @property
    def outputs(self):
        return self.codebook[:,1]

class MatchSearchTransfer():
    def __init__(self, codebook: PairedCodebook):
        self.codebook = codebook
    
    def transfer_sequence(self, target_sequence):
        with torch.no_grad():
            output_sequence = torch.stack([self.transfer(target) for target in target_sequence], dim=0)

        return output_sequence
    
    def transfer(self, target):
        _, opt_index = match_search(target.unsqueeze(0), self.codebook.targets)

        output = self.codebook.outputs[opt_index]
        return output

        
class AutoConcatenator():
    def __init__(self, corpus: PairedCorpus):
        self.corpus = corpus

    def quantize_transfer(self, input, target_latents, output_latents):

        differences = target_latents - input
        distances = torch.linalg.norm(differences, axis=(1, 2))
        opt_index = torch.argmin(distances)

        # cosine_sims = torch.nn.functional.cosine_similarity(input.unsqueeze(0), target_latents, dim=1)
        # cosine_norms = torch.mean(cosine_sims, axis=1)
        # opt_index = torch.argmax(cosine_norms)

        # xcorr_sims = torch.stack([self.xcorr_similarity(input, target) for target in target_latents])

        # opt_index = torch.argmax(xcorr_sims)

        # print("xxx")
        # print(opt_index)
        # print(xcorr_sims[opt_index])
        # print(torch.var_mean(target_latents[opt_index]))

        # print(distances[opt_index])

        output = output_latents[opt_index]

        return output
    
    def knn_transfer(self, input, target_latents, output_latents, k=1):

        # input = torch.nn.functional.normalize(input, dim=0)
        # target_latents = torch.nn.functional.normalize(target_latents, dim=1)

        differences = target_latents - input
        distances = torch.linalg.norm(differences, axis=(1, 2))
        k_opt_dist, k_opt_ind = torch.topk(-distances, k)
        normalized_dist_score = torch.nn.functional.softmax(k_opt_dist, dim=0)
        print(normalized_dist_score)

        # cosine_sims = torch.nn.functional.cosine_similarity(input.unsqueeze(0), target_latents, dim=1)
        # cosine_norms = torch.mean(cosine_sims, axis=1)
        # k_opt_dist, k_opt_ind = torch.topk(cosine_norms, k)
        # # normalized_dist_score = torch.nn.functional.softmax(k_opt_dist, dim=0)
        # normalized_dist_score = torch.ones_like(k_opt_dist)/k

        # normalized_dist_score = (1/distances)/torch.sum((1/k_opt_dist))
        # normalized_dist_score = torch.ones_like(k_opt_dist)/k

        transformed_components = [normalized_dist_score[i]*output_latents[j] for i, j in enumerate(k_opt_ind)]

        approximated_transform = torch.sum(torch.stack(transformed_components), dim=0)

        quantized_transform = self.quantize_transfer(approximated_transform, output_latents, output_latents)
        # quantized_transform = approximated_transform

        return quantized_transform

    
    def salt(self, input, target_latents, output_latents, max_steps=1, tolerance=1e-3):

        scaled_target_corpus = 1/max_steps * target_latents
        scaled_output_corpus = 1/max_steps * output_latents

        residual = input
        transformed_block = torch.zeros(input.shape, device=input.device)

        for i in range(max_steps):
            if torch.linalg.norm(residual) < tolerance:
                break

            differences = scaled_target_corpus - residual
            distances = torch.linalg.norm(differences, axis=(1, 2))
            min_index = torch.argmin(distances)

            best_fit_target = scaled_target_corpus[min_index]
            best_fit_output = scaled_output_corpus[min_index]

            transformed_block = transformed_block + best_fit_output
            residual = residual - best_fit_target

        return transformed_block
    
    def xcorr_similarity(self, x, y):
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        # print(x.shape)
        xcorr = torch.nn.functional.conv1d(x.unsqueeze(0), y.flip(-1).unsqueeze(1), padding=x.shape[1]-1, groups=x.shape[0])
        # print(xcorr.shape)
        mean_xcorr = torch.mean(xcorr, dim=2)
        # print(mean_xcorr.shape)
        xcorr_sim = torch.linalg.norm(mean_xcorr, dim=1)
        # print(xcorr_sim.shape)

        return xcorr_sim
    
    def autoconcat(self, input_waveform, codec, batch_size: int = 64, max_steps: int = 1, tolerance: float = 1e-3, noise=0, k=1):
        # input target waveform -> reconstructed output waveform
        target_latents = self.target_latents(codec, noise)
        output_latents = self.output_latents(codec)

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
                # transformed_embeddings = torch.cat((transformed_embeddings, self.salt(block, target_latents, output_latents, max_steps, tolerance).unsqueeze(0)), dim=0)
                # transformed_embeddings = torch.cat((transformed_embeddings, self.quantize_transfer(block, target_latents, output_latents).unsqueeze(0)), dim=0)
                transformed_embeddings = torch.cat((transformed_embeddings, self.knn_transfer(block, target_latents, output_latents, k).unsqueeze(0)), dim=0)

            batched_transformed_embeddings = batch_partition(transformed_embeddings, batch_size)
            batched_reconstruction = [codec.decode(batch) for batch in batched_transformed_embeddings]


        reconstructed_waveform = torch.cat(batched_reconstruction, dim=0).flatten()

        return reconstructed_waveform
    
    def target_latents(self, codec, noise = 0):
        latents = self.code_to_latents(self.corpus.target_blocks, codec)
        latents = latents + noise*torch.randn_like(latents)
        return latents
    
    def output_latents(self, codec):
        latents = self.code_to_latents(self.corpus.output_blocks, codec)
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

def match_search(input, codebook = None, distances = None):
    differences = codebook - input
    distances = torch.linalg.norm(differences, axis=(-2, -1))

    min_dist, opt_index = torch.min(distances, dim=-1)
    return min_dist, opt_index