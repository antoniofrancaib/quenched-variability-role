import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

L = np.pi
N = 512
w0 = -10
I_0 = 0.9
T = {'t_span': (0, 5000), 't_steps': 5000}

critical_w1 = 2 / derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0)
print(critical_w1)
w1 = critical_w1 + 0.2

epsilon = 0.005
perturbation = lambda theta: r_02(w0, I_0) + epsilon * np.cos(theta)

W = lambda delta_theta: w0 + w1 * np.cos(delta_theta)
delta_W =lambda theta: np.cos(theta)

ring = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=False)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ring.plot_state(ax[0])
ring.plot_dynamics(ax[1])

plt.tight_layout()
plt.show()

print(ring.calculate_bump_amplitude())