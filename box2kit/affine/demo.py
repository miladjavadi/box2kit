import dac
from box2kit.affine.model import AffineTransfer
from box2kit.utils import load_data as uload
import torchaudio
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    codec = dac.DAC.load(dac.utils.download()).to(DEVICE)
    sr = codec.sample_rate
    targets = codec(torch.cat(uload.load_dir("ins/beatbox", sr), axis=-1))[0].reshape(1024, -1).transpose(0,1).cpu().numpy()
    outputs = codec(torch.cat(uload.load_dir("ins/drum_kit", sr), axis=-1))[0].reshape(1024, -1).transpose(0,1).cpu().numpy()

    model = AffineTransfer(1024)
    model.fit(targets, outputs)

    test_seq = codec(uload.load_mono("ins/16.wav", sr))[0].reshape(1024, -1).transpose(0,1).cpu().numpy()
    transformed_test = model(test_seq)

    transformed_wave = codec.decode(torch.Tensor(transformed_test).to(DEVICE).squeeze(0))

    torchaudio.save("outs/ballzach.wav", transformed_wave.cpu().numpy())
    
if __name__ == "__main__":
    main()