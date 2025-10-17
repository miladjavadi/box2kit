import numpy as np
import torch
from box2kit.utils.load_data import batch_partition
from box2kit.utils.load_data import PairedWaveformDataset
from tqdm import tqdm

class PairedCodebook():
    """
    Codebook of paired target-output latent sequence pairs.

    Used for performing match search transfer.

    .. note::
            Both target- and output-domain coordinates should be provided for each point.
            Dim 1 in both the training_data and validation_data array corresponds to domain, with target-domain coordinates in [:, 0, ...], and output-domain coordinates in [:, 1, ...].

    Args:
        training_set (tensor[data pt., domain, latent dim., frame idx.]): Dataset of training sequence pairs.
        validation_set (tensor[data pt., domain, latent dim., frame idx.]): Dataset of validation sequence pairs. 
        codebook_length (int): Length of codebook/number of codewords.
        batch_size (int): Size of mini-batches for point-pair distance array calculations. Larger batches decrease computation time, but increase memory usage.
    """
    def __init__(self, training_set: torch.Tensor, validation_set: torch.Tensor, waveform_segment_length: int, codebook_length: int=512, batch_size: int=32):      
        if codebook_length < len(training_set):
            with torch.no_grad():
                self.codebook = self.greedy_codebook(training_set, validation_set, codebook_length, batch_size)
        else:
            raise ValueError(f"Desired codebook length exceeds number of training points ({codebook_length} > {len(training_set)}).")
        
        self.waveform_segment_length = waveform_segment_length
    
    def greedy_codebook(self, training_data, validation_data, codebook_length, batch_size):
        """
        Construct a best-fit codebook for quantizing a set of validation points from a subset of training points.

        Codebooks are fitted using a greedy algorithm, which minimizes expected target-domain Frobenius distance between codewords and validation points.

        Args:
            training_data (tensor[data pt., domain, latent dim., frame idx.]): Dataset of training sequence pairs.
            validation_data (tensor[data pt., domain, latent dim., frame idx.]): Dataset of validation sequence pairs. 
            codebook_length (int): Length of codebook/number of codewords.
            batch_size (int): Size of mini-batches for point-pair distance array calculations. Larger batches decrease computation time, but increase memory usage.
        
        Returns:
            codebook (tensor[codeword, domain, latent dim., frame idx.]): Constructed codebook.
        """
        codeword_indices = []

        point_pair_distances = self.point_pair_distance_array(training_data, validation_data, batch_size)
        for i in tqdm(range(codebook_length), desc="Codewords found"):
            new_codeword_index = self.greedy_search_step(codeword_indices, point_pair_distances)
            codeword_indices.append(new_codeword_index)

        codebook = training_data[codeword_indices]
        return codebook
    
    def greedy_search_step(self, codeword_indices: list[int], point_pair_distances: torch.Tensor):
        """
        Find codeword index which minimizes match search quantization error between new codebook and validation data.

        Args:
            codeword_indices (list of int): List of indices in current codebook
            point_pair_distances (Tensor[candidate pt., validation pt.]): Distances between all codeword candidate-validation point pairs

        Returns:
            best_index (int): Index of best performing codeword candidate
        """
        best_score = torch.inf

        codebook_distances = point_pair_distances[codeword_indices,:]

        codebook_candidate_distances = self.create_codebook_candidates(point_pair_distances, codebook_distances)
        _, best_index = self.batch_greedy_search(codebook_candidate_distances)

        return best_index
    
    def create_codebook_candidates(self, candidate_distances: torch.Tensor, codebook_distances: torch.Tensor) -> torch.Tensor:
        """
        Create array of point pair-distance matrices for new codebook candidates.

        Args:
            candidate_distances (Tensor[candidate pt., validation pt.]): Distances between candidate-validation point pairs.
            codebook_distances ([codeword pt., validation pt.]): Distances between existing codebook's codeword-validation point pairs.

        Returns:
            codebook_candidate_distances ([codebook candidate, codeword, validation point]): All codebook candidates' codeword-validation distance pairs.
        """
        n_candidates = candidate_distances.shape[0]
        expanded_codebook = codebook_distances.unsqueeze(0).expand(n_candidates, -1, -1)

        codebook_candidate_distances = torch.cat((expanded_codebook, candidate_distances.unsqueeze(1)), dim=1)

        return codebook_candidate_distances

    def batch_greedy_search(self, codebook_candidate_distances: torch.Tensor) -> tuple[float, int]:
        """
        Find best codebook candidate from an array of point-pair distance matrices.

        Args:
            codebook_candidate_distances (Tensor[codebook candidate, codeword, validation pt.]): All codebook candidates' codeword-validation distance pairs.

        Returns:
            best_score (float): Mean quantization error of best codebook candidate on validation data.
            best_codebook_index(int): Index of best codebook candidate.
        """
        min_distances, _ = torch.min(codebook_candidate_distances, dim=1)
        scores = torch.mean(min_distances, dim=1)
        best_score, best_codebook_index  = torch.min(scores, dim=0)

        return best_score, best_codebook_index
    
    def point_pair_distance_array(self, training_data: torch.Tensor, validation_data: torch.Tensor, batch_size: int):
        """
        Create array of point pair-distances between arrays of latent-coded training and validation frame sequences.

        Distance calculations are performed in mini-batches to reduce memory usage.
        
        Args:
            training_data (Tensor[training pt., domain, latent dim., frame idx.]): Training/codeword candidate sequences
            validation_data (Tensor[validation pt., domain, latent., frame idx.]): Validation sequences

        Returns:
            distances (Tensor[training pt., validation pt.]): Distances between training-validation point pairs
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
        """
        Append new codeword to codebook.

        Args:
            new_segment (Tensor[domain, latent dim., frame idx.]): Target and output domain-coordinates for new codeword.
        """
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
    """
    Perform match-search based timbre transfer using a pre-constructed codebook of latent sequence pairs.

    Args:
        codebook (PairedCodebook): Codebook used in match search.
    """
    def __init__(self, codebook: PairedCodebook):
        self.codebook = codebook
    
    def transfer_array(self, target_array, n=1):
        """
        Perform `n`th-order match-search based timbre transfer on an array of target latent sequences.

        Target latent sequences are matched with their `n` nearest neighboring codewords.
        Output latent sequences are generated as the mean of the matching codewords' output-domain counterparts, re-quantized to the nearest output-domain codeword.

        Args:
            target_array (Tensor[array idx., latent dim., frame idx.]): Array of target latent sequences.
            n (int): Match search order.
        
        Returns:
            output_array (Tensor[array idx., latent dim., frame idx.]): Generated array of output latent sequences.
        """
        with torch.no_grad():
            output_array = torch.cat([self.transfer(target, n) for target in target_array], dim=0)

        return output_array
    
    def transfer(self, target, n=1):
        """
        Perform `n`th-order match-search based timbre transfer on a target latent sequence.

        Target latent sequences are matched with their `n` nearest neighboring codewords.
        Output latent sequences are generated as the mean of the matching codewords' output-domain counterparts, re-quantized to the nearest output-domain codeword.

        Args:
            target (Tensor[latent dim., frame idx.]): Target latent sequence.
            n (int): Match search order.
        
        Returns:
            output (Tensor[latent dim., frame idx.]): Generated output latent sequence.
        """
        _, opt_indices = match_search(target.unsqueeze(0), self.codebook.targets, n)

        matched_outputs = self.codebook.outputs[opt_indices,:,:]
        mean_output = torch.mean(matched_outputs, dim=0)

        _, opt_index = match_search(mean_output.unsqueeze(0), self.codebook.outputs, 1)

        output = self.codebook.outputs[opt_index]

        return output


def match_search(input, codebook, n):
    """
    Match an input latent sequence with its `n` nearest neighboring entry in a codebook.

    Inputs are matched with the codebook entries nearest to them in Frobenius distance.

    Args:
        input (Tensor[latent dim., frame idx.]): Input latent sequence.
        codebook (Tensor[codeword, latent dim., frame idx.]): Codebook to match against.

    Returns:
        min_dist (Tensor): Frobenius distance between input and `n` nearest neighboring codewords.
        opt_index (Tensor): Index in codebook of `n` nearest neighboring codeword.
    """
    differences = codebook - input
    distances = torch.linalg.norm(differences, axis=(-2, -1))

    min_dists, opt_indices = torch.sort(distances, dim=-1)
    return min_dists[:n], opt_indices[:n]