import dac
import torchaudio
import torch
from box2kit.utils.load_data import load_mono

codec = dac.DAC.load(dac.utils.download()).to("cuda")

# # zeros = torch.zeros(1,1024,2000).to("cuda")
# noise = 10*torch.randn(1,1024,1000).to("cuda")
# noise = codec.quantizer(noise)[0]
# test_audio = codec.decode(noise)

# latents = codec.encode(load_mono("training_data/drum_kit/1.wav", codec.sample_rate).unsqueeze(0).to("cuda")[:,:,:int(4*codec.sample_rate)])[0]
# latents = codec.quantizer(latents)[0]
# test_audio = codec.decode(-latents)

# torchaudio.save("outs/anti_drums.wav", test_audio[0].cpu().detach(), codec.sample_rate)

silence = torch.zeros(1,1,2*codec.sample_rate).to("cuda")

latents = codec.encode(silence)[0]
print(torch.mean(latents, dim=-1))
test_audio = codec.decode(latents)
torchaudio.save("outs/radio_silence.wav", test_audio[0].cpu().detach(), codec.sample_rate)
