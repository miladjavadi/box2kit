import torch

class MultiScaleSpectralDistance(torch.nn.Module):
    def __init__(self, window_lengths: list[int]):
        super().__init__()
        self.windows_lengths = window_lengths

    def spectral_distance(self, x, y, window_length):
        x_amp = self.spec_amp(x, window_length)
        y_amp = self.spec_amp(y, window_length)

        spectral_distance = (torch.linalg.matrix_norm(x_amp - y_amp) / torch.linalg.matrix_norm(x_amp))
        + torch.log(torch.sum(torch.abs(x_amp - y_amp), (-2, -1)))

        return spectral_distance

    def spec_amp(self, x, window_length):
        return torch.abs(torch.stft(x, window_length, return_complex=True))

    def forward(self, x, y):
        distances = torch.zeros(x.shape[0])
        for window_length in self.window_lengths:
            distances += self.spectral_distance(x, y)
        
        mean_distance = torch.mean(distances)
        return mean_distance