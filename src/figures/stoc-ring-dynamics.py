import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from src.utils.common_utils import Ring, derivative_nonlinearity, r_02

import numpy as np

L = np.pi
N = 256
w0 = -10
I_0 = 0.9

A = 10.0
B = 0
C = 0.0

T = {'t_span': (0, 5000), 't_steps': 5000}

r_0 = r_02(w0, I_0)
R_1 = np.sqrt((2*np.pi/(N))*(A+(C/2)))

critical_w1 = 2 / derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0) - R_1
w1 = critical_w1 - 0.2

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()
W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 

ring = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=True)

print("(stochastic) The amplitude of the final activity profile is:", ring.calculate_bump_amplitude(filter_noise=True))

ring2 = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=False)

print("(deterministic) The amplitude of the final activity profile is:", ring2.calculate_bump_amplitude())

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

ring.plot_state(axes[0, 0], timestep=0)
axes[0, 0].set_title('State at timestep 0 (with variability)')
ring.plot_state(axes[0, 1], timestep=-1)
axes[0, 1].set_title('State at final timestep (with variability)')
ring.plot_dynamics(axes[0, 2])
axes[0, 2].set_title('Dynamics (with variability)')


ring2.plot_state(axes[1, 0], timestep=0)
axes[1, 0].set_title('State at timestep 0 (without variability)')
ring2.plot_state(axes[1, 1], timestep=-1)
axes[1, 1].set_title('State at final timestep (without variability)')
ring2.plot_dynamics(axes[1, 2])
axes[1, 2].set_title('Dynamics (without variability)')

plt.tight_layout()
plt.show()



"""
k = 3
k_smallest_psi_1, k_smallest_theta = ring.find_k_smallest_psi_1(k)
k_greatest_R_1, k_greatest_theta = ring.find_k_greatest_R_1(k)

print(f"The {k} smallest psi_1 values and their corresponding theta values are:")
for i in range(k):
    print(f"psi_1: {k_smallest_psi_1[i]}, theta: {k_smallest_theta[i]}")

print(f"The {k} greatest R_1 values and their corresponding theta values are:")
for i in range(k):
    print(f"psi_1: {k_greatest_R_1[i]}, theta: {k_greatest_theta[i]}")

    
for theta in k_smallest_theta:
    axes[0, 0].axvline(x=theta, color='green', linestyle='--', label='k smallest theta' if theta == k_smallest_theta[0] else "")
    axes[0, 1].axvline(x=theta, color='green', linestyle='--')


for theta in k_greatest_theta:
    axes[0, 0].axvline(x=theta, color='blue', linestyle='--', label='k smallest theta' if theta == k_smallest_theta[0] else "")
    axes[0, 1].axvline(x=theta, color='blue', linestyle='--')

axes[0, 1].axvline(x=np.mean(k_smallest_theta), color='red', linestyle='--')
axes[0, 1].axvline(x=np.mean(k_greatest_theta), color='black', linestyle='--')
"""