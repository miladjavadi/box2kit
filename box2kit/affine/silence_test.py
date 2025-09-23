import dac
import torchaudio
import torch

codec = dac.DAC.load(dac.utils.download()).to("cuda")

# zeros = torch.zeros(1,1024,2000).to("cuda")
noise = 0.5*torch.randn(1,1024,1000).to("cuda")
test_audio = codec.decode(noise)

torchaudio.save("outs/noise_5e-2.wav", test_audio[0].cpu().detach(), codec.sample_rate)