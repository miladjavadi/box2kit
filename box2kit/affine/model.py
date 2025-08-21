import torch
from torch import nn
import pytorch_lightning as pl
import dac
from dac import DAC
from audiotools import AudioSignal
from box2kit.gantransfer.ganmodel import DiscriminatorV2
from torch import optim
from pytorch_lightning.callbacks import Callback
import torchaudio
from box2kit.utils.load_data import load_mono, reshape_data
import os
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
# from sklearn.multioutput import MultiOutputRegressor

class AffineLightning(pl.LightningModule):
    def __init__(self,
                 input_block_length: int,
                 output_block_lengths: int,
                 nframes: int,
                 spectrum_dims: list[int],
                 nfft: int = None,
                 lambda_embedding: float = 1,
                 lambda_adversarial: float = 1,
                 codec: DAC = DAC.load(dac.utils.download()),
                 spectral_loss_fn = dac.nn.loss.MultiScaleSTFTLoss([2048, 1024, 512, 256, 128, 64]),
                 warmup: int = 250,
                 lr: float = 1e-4):
        super().__init__()

        self.generator, self.discriminator = self.initialize_models(spectrum_dims, nfft)
        self.codec = codec
        self.sr = codec.sample_rate
        self.lr = lr

        self.spectral_loss_fn = spectral_loss_fn
        self.embedding_loss_fn = nn.MSELoss()
        self.adversarial_loss_fn = nn.BCELoss()

        self.lambda_embedding = lambda_embedding
        self.lambda_adversarial = lambda_adversarial
        self.warmup = warmup

        self.input_block_length = input_block_length
        self.output_block_length = output_block_lengths
        self.nframes = nframes

        self.adversarial_phase = False
        # the objective of the discriminator is to return 1 for real target recordings, and 0 for synthesized ones

        self.automatic_optimization = False

        self.real_label = 1
        self.fake_label = 0

        self.codec.eval()
        self.codec.requires_grad_(False)

        self.save_hyperparameters(ignore=["codec", "spectral_loss_fn"])
    
    def initialize_models(self, spectrum_dims: list[int], nfft: int = None):
        generator = AffineTransfer(1024)
        discriminator = DiscriminatorV2(spectrum_dims, nfft=nfft)
        return generator, discriminator
    
    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        input_waveforms = batch
        query, target = input_waveforms

        gen_optimizer, discr_optimizer = self.optimizers()

        with torch.no_grad():
            Z_query = self.codec.encode(query)[0]
            Z_target = self.codec.encode(target)[0]

        # train generator
        self.toggle_optimizer(gen_optimizer)

        Z_gen = self.generator(Z_query)

        with torch.no_grad():
            gen = self.codec.decode(Z_gen)
            target_post = self.codec.decode(Z_target) # use post-dac target to match dims
            stft_gen = torch.stft(gen.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=gen.device), return_complex=True).abs()
            stft_target = torch.stft(target_post.squeeze(1), self.discriminator.nfft, window=torch.hann_window(self.discriminator.nfft, device=target_post.device), return_complex=True).abs()
        
        # calculate generator losses
        gen_optimizer.zero_grad()

        spectral_loss = self.spectral_loss_fn(AudioSignal(gen, self.sr), AudioSignal(target_post, self.sr))
        embedding_loss = self.embedding_loss_fn(Z_gen, Z_target)

        if self.adversarial_phase:
            gen_score = self.discriminator(stft_gen)
            real_labels = torch.full_like(gen_score, fill_value=self.real_label)
            # how convinced the discriminator is that generated waveforms are real
            adversarial_loss = self.adversarial_loss_fn(gen_score, real_labels)

        else:
            adversarial_loss = 0

        generator_loss = spectral_loss + self.lambda_embedding * embedding_loss + self.lambda_adversarial * adversarial_loss
        self.manual_backward(generator_loss)
        gen_optimizer.step()
        self.untoggle_optimizer(gen_optimizer)

        # train discriminator
        if self.adversarial_phase:
            self.toggle_optimizer(discr_optimizer)
            discr_optimizer.zero_grad()
            # how convinced the discriminator is that target waveforms are real, and generated waveforms are fake
            real_score = self.discriminator(stft_target)
            gen_score = self.discriminator(stft_gen.detach())

            real_labels = torch.full_like(real_score, fill_value=self.real_label)
            fake_labels = torch.full_like(gen_score, fill_value=self.fake_label)

            discr_loss = self.adversarial_loss_fn(real_score, real_labels) + self.adversarial_loss_fn(gen_score, fake_labels)
            self.manual_backward(discr_loss)
            discr_optimizer.step()
            self.untoggle_optimizer(discr_optimizer)
        else:
            discr_loss = 0
        
        self.log("d_loss", discr_loss, prog_bar=True, logger=True)
        self.log("g_loss", generator_loss, prog_bar=True, logger=True)
        self.log("spectral_loss", spectral_loss, prog_bar=True, logger=True)
        self.log("embedding_loss", embedding_loss, prog_bar=True, logger=True)
        self.log("adversarial_loss", adversarial_loss, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        discr_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [gen_optimizer, discr_optimizer], []
    
    def on_train_epoch_start(self):
        self.codec.eval()
        self.adversarial_phase = True if self.current_epoch >= self.warmup else False
        return super().on_train_epoch_start()

    def on_fit_epoch_start(self):
        self.codec.eval()
    
    def on_validation_epoch_start(self):
        self.codec.eval()
    

class AffineTransferOld(nn.Module):
    def __init__(self,
                 input_dim: int = 1024):
        super().__init__()

        self.A = nn.Parameter(torch.eye(input_dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(input_dim, 1, dtype=torch.float32))

    def forward(self, x):
            y_hat = self.A @ x + self.b # Ax == Ax[:,i] for columns i in x
            return y_hat
    
class GenerationCallback(Callback):
    def __init__(self, block_length: int, test_file: str, out_dir: str, test_freq: int = 5):
        self.test_file = test_file
        self.test_freq = test_freq
        self.out_dir = out_dir
        self.block_length = block_length

        self.output_test = self.test_file is not None and self.out_dir is not None

    def on_train_start(self, trainer, pl_module):
        if self.output_test:
            try:
                os.mkdir(self.out_dir)
            except FileExistsError:
                pass
        return super().on_train_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.trainer.current_epoch
        if epoch % self.test_freq == 0 and self.output_test:
            test_wave = [load_mono(self.test_file, pl_module.sr)]
            test_segs = reshape_data(test_wave, pl_module.input_block_length).to(pl_module.device)
            with torch.inference_mode():
                test_frames = pl_module.codec.encode(test_segs)[0]
                reconstructed_wave = torch.cat([pl_module.codec.decode(pl_module(seg.unsqueeze(0))) for seg in test_frames], dim=2).squeeze(0)

            torchaudio.save(f"{self.out_dir}/epoch_{epoch}.wav", reconstructed_wave.cpu(), pl_module.sr)
        return super().on_train_epoch_end(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module: AffineLightning):
        if self.output_test:
            test_wave = [load_mono(self.test_file, pl_module.sr)]
            test_segs = reshape_data(test_wave, pl_module.input_block_length).to(pl_module.device)
            with torch.inference_mode():
                test_frames = pl_module.codec.encode(test_segs)[0]
                reconstructed_wave = torch.cat([pl_module.codec.decode(pl_module(seg.unsqueeze(0))) for seg in test_frames], dim=2).squeeze(0)

            torchaudio.save(f"{self.out_dir}/epoch_{pl_module.trainer.current_epoch}.wav", reconstructed_wave.cpu(), pl_module.sr)
        return super().on_train_end(trainer, pl_module)
    
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.codec.eval()
        pl_module.adversarial_phase = True if trainer.current_epoch >= pl_module.warmup else False
        return super().on_train_epoch_start(trainer, pl_module)

    # def on_fit_epoch_start(self):
    #     self.codec.eval()
    
    # def on_validation_epoch_start(self):
    #     self.codec.eval()

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
    
    def fit(self, targets, outputs, n_trials = 100, threshold = 10, n_samples = 256):
        n_points = targets.shape[0]

        bestimator = self.estimator
        if self.is_fitted:
            approximations = bestimator.predict(targets)
            residuals = approximations - outputs
            distances = np.linalg.norm(residuals, axis=0)
            best_score = np.count_nonzero(np.less(distances, threshold))/n_points # ratio of inliers
        else:
            best_score = 0

        rng = np.random.default_rng()
        for i in range(n_trials):
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
        
        self.estimator = bestimator
        self.is_fitted = True
    
    def __call__(self, targets):
        return self.estimator.predict(targets)

    # def transform_sequence(self, targets, A, b):
    #     transformed = np.stack([self.transform_vector(target, A, b) for target in targets], axis=0)
    #     return transformed
    
    # def transform_vector(self, target, A, b):
    #     transformed = A @ target + b
    #     return transformed
    
    
if __name__ == "__main__":
    x = np.random.rand(300, 1024)
    y = np.random.rand(300, 1024)
    model = AffineTransfer(1024)
    model.fit(x, y)

    y = model(x)
    print(x.shape, y.shape, model.estimator.coef_.shape, model.estimator.intercept_.shape)