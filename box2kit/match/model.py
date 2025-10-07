import numpy as np
import torch
from box2kit.utils.load_data import batch_partition
from box2kit.utils.load_data import PairedWaveformDataset
from tqdm import tqdm

class PairedCodebook():
    def __init__(self, training_set: torch.Tensor, validation_set: torch.Tensor, waveform_segment_length: int, codebook_length: int=512, batch_size: int=32):      
        if codebook_length < len(training_set):
            with torch.no_grad():
                self.codebook = self.greedy_codebook(training_set, validation_set, codebook_length, batch_size)
        else:
            raise ValueError(f"Desired codebook length exceeds number of training points ({codebook_length} > {len(training_set)}).")
        
        self.waveform_segment_length = waveform_segment_length
    
    def greedy_codebook(self, training_data, validation_data, codebook_length, batch_size):
        codeword_indices = []

        point_pair_distances = self.point_pair_distance_array(training_data, validation_data, batch_size)
        for i in tqdm(range(codebook_length), desc="Codewords found"):
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

        codebook_distances = point_pair_distances[codeword_indices,:]

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
            start = i
            stop = i + batch_size
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
    
    def transfer_sequence(self, target_sequence, n=1):
        with torch.no_grad():
            output_sequence = torch.stack([self.transfer(target) for target in target_sequence], dim=0)

        return output_sequence
    
    def transfer(self, target, n=1):
        _, opt_indices = match_search(target.unsqueeze(0), self.codebook.targets, n)

        matched_outputs = self.codebook.outputs[opt_indices,:,:]

        _, opt_index = match_search(matched_outputs.unsqueeze(0), self.codebook.outputs, 1)

        output = self.codebook.outputs[opt_index]

        return output


def match_search(input, codebook, n=1):
    differences = codebook - input
    distances = torch.linalg.norm(differences, axis=(-2, -1))

    min_dist, opt_index = torch.min(distances, dim=-1)
    return min_dist, opt_index