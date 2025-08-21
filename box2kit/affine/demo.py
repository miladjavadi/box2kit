import dac
from box2kit.affine.model import AffineTransfer
from box2kit.utils import load_data as uload
import torchaudio
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    sr = codec.sample_rate
    targets = uload.safe_encode(uload.reshape_data(uload.load_dir("training_data/beatbox", sr)[:2], 700),codec,8).transpose(1,2).reshape(-1,1024).cpu().numpy()
    outputs = uload.safe_encode(uload.reshape_data(uload.load_dir("training_data/drum_kit", sr)[:2], 700),codec,8).transpose(1,2).reshape(-1,1024).cpu().numpy()

    model = AffineTransfer(1024)
    model.fit(targets, outputs, threshold=100, n_samples=200)

    test_seq = uload.safe_encode(uload.reshape_data([uload.load_mono("16.wav", sr)], 700),codec,8).transpose(1,2).reshape(-1,1024).cpu().numpy()
    transformed_test = model(test_seq)

    transformed_wave = uload.safe_decode(torch.Tensor(transformed_test).to(DEVICE).transpose(0,1).reshape(-1, 1024, 700), codec, 8).reshape(1,-1)

    torchaudio.save("outs/ballzach.wav", transformed_wave.cpu().numpy())

if __name__ == "__main__":
    main()