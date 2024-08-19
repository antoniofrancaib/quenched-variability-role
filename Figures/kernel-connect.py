import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02, multi_peak_perturbation

import numpy as np

L = np.pi
N = 64
w0 = -10
I_0 = 0.9

A = 1
B = 0
C = 1

T = {'t_span': (0, 5000), 't_steps': 5000}

r_0 = r_02(w0, I_0)
R_1 = np.sqrt((np.pi * (2*A+C) / (4*N)))
critical_w1 = 2 / derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0) - R_1
w1 = 5

phases = [3*np.pi/4] #-3*np.pi/4, 3*np.pi/4
epsilon = 0.005
perturbation = lambda theta: multi_peak_perturbation(theta, r_0, phases, epsilon)

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()
W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 

ring = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=True)

plt.figure(figsize=(10, 6))

plt.plot(ring.theta, W(ring.theta), color='red', linewidth=4, label='Connectivity Kernel (No Variability)')

for j in range(N):
    plt.plot(ring.theta, W(ring.theta) + ring.quenched_variability[j], color='gray', alpha=0.5, linestyle='--', label='Connectivity with Quenched Variability' if j == 0 else "")

plt.legend(loc='best', fontsize='small')

plt.xlabel('$\\theta$')
plt.ylabel('Connectivity')
plt.title('Connectivity Kernel with and without Quenched Variability')

plt.grid(True)
plt.show()