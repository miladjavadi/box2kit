import torch
from torch import nn
import pytorch_lightning as pl
import dac
from dac import DAC
from audiotools import AudioSignal
from torch import optim
from pytorch_lightning.callbacks import Callback
import torchaudio
from box2kit.utils.load_data import load_mono, reshape_data
from box2kit.utils.constants import DAC_SILENCE
import os
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from tqdm import tqdm

class AffineTransfer():
    def __init__(self, input_dim: int = 1024):
        """
        here, latent sequences are arranged [T, D], which is a transpose of how it is in the paper,
        and other model implementations. this is because each time step is considered its own data point,
        making it more intuitive for time to be on the first axis.
        """ 

        self.dim = input_dim
        self.estimator = LinearRegression()
        self.is_fitted = False
    
    def fit(self, targets, outputs, n_trials = 100, threshold = 10, n_samples = 256, trim_silence = True, gate_threshold = 140):
        # remove silent frames by masking frames that are far enough away from the "silence region" in latent space
        if trim_silence:
            mask = np.linalg.norm(targets - np.asarray(DAC_SILENCE).reshape(1, -1), axis=1) > gate_threshold
            targets = targets[mask]
            outputs = outputs[mask]

        n_points = targets.shape[0]

        bestimator = self.estimator
        if self.is_fitted:
            approximations = bestimator.predict(targets)
            residuals = approximations - outputs
            distances = np.linalg.norm(residuals, axis=0)
            best_score = np.count_nonzero(np.less(distances, threshold))/n_points # ratio of inliers to total points
        else:
            best_score = 0

        rng = np.random.default_rng()
        for i in tqdm(range(n_trials), desc="RANSAC Trials"):
            model = clone(bestimator) # new instance of lin regressor
            sample_indices = rng.choice(n_points, n_samples)
            target_samples = targets[sample_indices]; output_samples = outputs[sample_indices]
            model.fit(target_samples, output_samples)

            approximations = model.predict(targets)
            residuals = approximations - outputs
            distances = np.linalg.norm(residuals, axis=0)
            score = np.count_nonzero(np.less(distances, threshold))/n_points

            if score > best_score:
                bestimator = model
                best_score = score
                self.is_fitted = True
        
        self.estimator = bestimator
        if best_score == 0:
            print("No sufficient estimator found.")
    
    def __call__(self, targets):
        return self.estimator.predict(targets)
    
    
if __name__ == "__main__":
    x = np.random.rand(300, 1024)
    y = np.random.rand(300, 1024)
    model = AffineTransfer(1024)
    model.fit(x, y)

    y = model(x)
    print(x.shape, y.shape, model.estimator.coef_.shape, model.estimator.intercept_.shape)
