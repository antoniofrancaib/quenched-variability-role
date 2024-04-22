import numpy as np
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

L = np.pi
N = 256
w0 = -1
I_0 = 0.1

critical_w1 = 2 / derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0)
w1 = critical_w1 - 0.5

epsilon = 0.015
perturbation = lambda theta: r_02(w0, I_0) + epsilon * np.cos(theta)
W = lambda delta_theta: w0 + w1 * np.cos(delta_theta)

ring = Ring(L, N, W, I_0, perturbation)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ring.plot_state(ax[0])
ring.plot_dynamics(ax[1])

plt.tight_layout()
plt.show()
