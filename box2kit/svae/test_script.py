from box2kit.svae.rave_pqmf import PQMF
from box2kit.utils import load_data
import torchaudio

filename="16.wav"

waveform = load_data.load_mono("16.wav", 48000).unsqueeze(0)

pqmf = PQMF(80, 16)

recon_waveform = pqmf.inverse(pqmf(waveform))

torchaudio.save("sanity_test_rave.wav", recon_waveform.squeeze(0), 48000)