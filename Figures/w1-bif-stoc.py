import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02


L = np.pi
N = 64

T = {'t_span': (0, 5000), 't_steps': 5000}

w0 = -10
I_0 = 0.9

A = 100.0
B = 0.0 
C = 0.0

R = np.sqrt((2*np.pi/N)*(A+(C/2)))
R_1 = np.sqrt((np.pi * (2*A+C) / (4*N)))

r_0 = r_02(w0, I_0)
critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R_1
wout_critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) 
print(critical_w1)

amplitudes = []
delta = 0.5
w1_values = np.linspace(1, 12, 5)

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
    
delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()

num_simulations = 10
all_simulation_amplitudes = []


for w1 in w1_values:
    simulation_amplitudes = []
    for _ in range(num_simulations):
        W = lambda delta_theta: w0 + w1 * np.cos(delta_theta)
        ring = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=True)  
        amplitude = ring.calculate_bump_amplitude()  
        simulation_amplitudes.append(amplitude)
    
    all_simulation_amplitudes.append(simulation_amplitudes)

amplitude_means = np.array([np.mean(simulation_amplitudes) for simulation_amplitudes in all_simulation_amplitudes])
amplitude_stds = np.array([np.std(simulation_amplitudes) for simulation_amplitudes in all_simulation_amplitudes])  

confidence_interval = stats.norm.interval(0.95, loc=amplitude_means, scale=amplitude_stds / np.sqrt(num_simulations))

lower_bounds, upper_bounds = confidence_interval

plt.figure(figsize=(10, 6))
plt.plot(w1_values, amplitude_means, 'ro', markersize=5, label='Mean Amplitude')

plt.errorbar(w1_values, amplitude_means, yerr=[amplitude_means - lower_bounds, upper_bounds - amplitude_means], fmt='none', ecolor='black', capsize=5, label='95% Confidence Interval')

plt.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical W1: {critical_w1:.2f}')  
plt.axvline(x=wout_critical_w1, color='r', linestyle='--', label=f'Critical W2: {wout_critical_w1:.2f}')  

plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title(f'Bifurcation Diagram for $W_0=${w0} and $I_0=${I_0}') 
plt.grid(True)  
plt.legend() 
plt.show()
