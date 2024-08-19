import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import random
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

L = np.pi
N = 256

T = {'t_span': (0, 5000), 't_steps': 5000}

w0 = -20
I_0 = 1.5

r_0 = r_02(w0, I_0)
critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0)

amplitudes = []
delta = 2
w1_values = np.linspace(critical_w1-delta, critical_w1+delta, 100)

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

delta_W = 0

for w1 in w1_values:
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 
    ring = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=False)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)

plt.figure(figsize=(10, 6))  
plt.plot(w1_values, amplitudes, 'ro', markersize=2)  
plt.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical W1: {critical_w1:.2f}')  
plt.xlabel('$W_1$')  
plt.ylabel('Bump Amplitude') 
plt.title(f'Bifurcation Diagram for $W_0=${w0} and $I_0=${I_0}') 
plt.grid(True)  
plt.legend() 
plt.show()  
