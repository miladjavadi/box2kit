from box2kit.svae.model import TransferGAN
import numpy as np

model = TransferGAN.load_from_checkpoint("outs/trialz.ckpt")

print(model.model.prior.means, model.model.prior.log_vars)

np.savetxt("outs/means.csv", model.model.prior.means.detach().numpy())
np.savetxt("outs/vars.csv", model.model.prior.log_vars.detach().numpy())