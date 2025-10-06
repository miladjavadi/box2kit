import numpy as np
from matplotlib import pyplot as plt

def normal_pdf(mu, std):
    return 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * std**2))

x = np.linspace(-2,2)
y = normal_pdf(1.15,0.25)


print(y[np.where(x>=1.05)])
# plt.plot(x, y)
# plt.show()

# np.savetxt("outs/posterior2.dat", np.column_stack((x, y)), fmt="%.6f")