import numpy as np

class PairedCorpus():
    def __init__(self, query_frames, target_frames):
        if query_frames.ndim != 3 or target_frames.ndim != 3:
            raise ValueError("Dataset arrays must be three-dimensional (N x D x T).")
    
        self.query_frames = query_frames # N x D x T
        self.target_frames = target_frames # N x D x T
    
    def append(self, new_query_frames, new_target_frames) -> None:
        if new_query_frames.shape != new_target_frames:
            raise ValueError(f"Paired query frame and target frame dimensions do not match ({new_query_frames.shape} and {new_target_frames.shape}).")
        
        try:
            self.query_frames = np.concat((self.query_frames, new_query_frames), axis=0)
        except ValueError:
            raise ValueError(f"New query frames must match dimensionality with existing corpus in all dimension except 0 ({self.query_frames.shape} and {new_query_frames.shape}).")
        
        try:
            self.target_frames = np.concat((self.target_frames, new_target_frames), axis=0)
        except ValueError:
            raise ValueError(f"New target frames must match dimensionality with existing corpus in all dimension except 0 ({self.target_frames.shape} and {new_target_frames.shape}).")
    
    @property
    def nframes(self):
        return self.query_frames.shape[0]
    
    @property
    def ndims(self):
        return self.query_frames.shape[1]
    
    @property
    def frame_length(self):
        return self.query_frames.shape[2]

        
class AutoConcatenator():
    def __init__(self, corpus: PairedCorpus):
        self.corpus = corpus

    def quantize_transfer(self, input):
        differences = self.corpus.query_frames - input
        distances = np.linalg.norm(differences, axis=(1, 2))
        min_index = np.argmin(distances)

        output = self.corpus.target_frames[min_index]

        return output