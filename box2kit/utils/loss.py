import torch

class MultiScaleSpectralDistance(torch.nn.Module):
    def __init__(self, window_lengths: list[int]):
        super().__init__()
        self.window_lengths = window_lengths

    def spectral_distance(self, x, y, window_length):
        x_amp = self.spec_amp(x, window_length)
        y_amp = self.spec_amp(y, window_length)

        spectral_distance = (torch.linalg.matrix_norm(x_amp - y_amp) / torch.linalg.matrix_norm(x_amp))
        + torch.log(torch.sum(torch.abs(x_amp - y_amp), (-2, -1)))

        return spectral_distance

    def spec_amp(self, x, window_length):
        return torch.abs(torch.stft(x.squeeze(1), window_length, window=torch.hann_window(window_length, device=x.device), return_complex=True))

    def forward(self, x, y):
        distances = torch.zeros(x.shape[0]).to(x.device)
        for window_length in self.window_lengths:
            distances += self.spectral_distance(x, y, window_length)
        
        mean_distance = torch.mean(distances)
        return mean_distance