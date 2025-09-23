import dac
import torchaudio
import torch
from box2kit.utils.load_data import load_mono
from box2kit.utils.constants import DAC_SILENCE

codec = dac.DAC.load(dac.utils.download()).to("cuda")

# # zeros = torch.zeros(1,1024,2000).to("cuda")
# noise = 10*torch.randn(1,1024,1000).to("cuda")
# noise = codec.quantizer(noise)[0]
# test_audio = codec.decode(noise)

latents = codec.encode(load_mono("training_data/beatbox/8.wav", codec.sample_rate).unsqueeze(0).to("cuda")[:,:,:int(4*codec.sample_rate)])[0]
# latents = codec.quantizer(latents)[0]
# test_audio = codec.decode(-latents)

# zeros = torch.zeros(1,1024,2000).to("cuda")
# noisy_silence = (torch.tensor(DAC_SILENCE).view(1, -1, 1) + 25*torch.randn(1,1024,1000)).to("cuda")
# noise = codec.quantizer(noisy_silence)[0]
# test_audio = codec.decode(noisy_silence)

mask = torch.linalg.vector_norm(latents-torch.tensor(DAC_SILENCE).view(1, -1, 1).to("cuda"), dim=1)[0] > 200

trunc_latents = latents[:,:,mask]
print(trunc_latents.shape[-1]/latents.shape[-1])

test_audio = codec.decode(trunc_latents)

torchaudio.save("outs/trunc_drums.wav", test_audio[0].cpu().detach(), codec.sample_rate)

# silence = torch.zeros(1,1,2*codec.sample_rate).to("cuda")

# latents = codec.encode(silence)[0]
# torch.set_printoptions(threshold=10_000)
# print(torch.mean(latents, dim=-1))
# test_audio = codec.decode(latents)
# torchaudio.save("outs/radio_silence.wav", test_audio[0].cpu().detach(), codec.sample_rate)
