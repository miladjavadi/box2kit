import dac
import torchaudio
import torch

codec = dac.DAC.load(dac.utils.download()).to("cpu")

zeros = torch.zeros(1,1024,10000).to("cpu")
test_audio = codec.decode(zeros)

torchaudio.save("outs/silence.wav", test_audio[0].cpu().detach(), codec.sample_rate)