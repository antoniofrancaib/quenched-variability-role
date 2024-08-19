import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from src.utils.common_utils import Ring, derivative_nonlinearity, r_02
from scipy.stats import norm, sem


L = np.pi
N = 64 # change N to 64 everywhere, the difference is bigger, cite paper Alex
T = {'t_span': (0, 5000), 't_steps': 5000}

w0 = -30
I_0 = 2

r_0 = r_02(w0, I_0)
critical_w1_without = 2 / derivative_nonlinearity(w0 * r_0 + I_0)

delta = 2
num_sim = 50
w1_values = np.linspace(critical_w1_without - delta, critical_w1_without + delta, num_sim)

amplitudes_without = []

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

A = 1.0
B = 0.0
C = 0.0

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()

for w1 in w1_values:
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 
    ring_without = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability = False)
    amplitude_without = ring_without.calculate_bump_amplitude()
    amplitudes_without.append(amplitude_without)

# R = np.sqrt((2*np.pi/N)*(A+(C/2)))
R_1 = np.sqrt((np.pi * (2*A+C) / (4*N)))

critical_w1_with = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R_1

num_simulations = 20
all_simulation_amplitudes = []

for w1 in w1_values:
    simulation_amplitudes = []
    for _ in range(num_simulations):
        W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 
        ring_with = Ring(L, T, N, W, delta_W, I_0, perturbation, use_quenched_variability=True)
        amplitude = ring_with.calculate_bump_amplitude(filter_noise=True)
        simulation_amplitudes.append(amplitude)
    
    all_simulation_amplitudes.append(simulation_amplitudes)


amplitude_means = np.array([np.mean(sim) for sim in all_simulation_amplitudes])
threshold = 0.5
amplitude_means[amplitude_means < threshold] = 0

# Calculate standard deviations and confidence intervals
amplitude_stds = np.array([np.std(sim) for sim in all_simulation_amplitudes])
confidence_interval = norm.interval(0.95, loc=amplitude_means, scale=amplitude_stds / np.sqrt(num_simulations))
lower_bounds, upper_bounds = confidence_interval

plt.figure(figsize=(12, 8))
plt.plot(w1_values, amplitudes_without, 'r-', label='No Variability')

# Plot with conditional error bars
for i, (x, mean, lower, upper) in enumerate(zip(w1_values, amplitude_means, lower_bounds, upper_bounds)):
    if mean == 0:
        plt.plot(x, mean, 'bo')  # Just the dot
    else:
        plt.errorbar(x, mean, yerr=[[mean - lower], [upper - mean]], fmt='bo', ecolor='gray', capsize=5)

plt.axvline(x=critical_w1_with, color='b', linestyle='--', label=f'Critical W1 with Variability: {critical_w1_with:.2f}')
plt.axvline(x=critical_w1_without, color='r', linestyle='--', label=f'Critical W1 without Variability: {critical_w1_without:.2f}')
plt.xlabel('$W_1$')
plt.ylabel('Amplitude')
plt.title('Amplitude vs. $w_1$ with and without Quenched Variability')
plt.grid(True)
plt.legend()
plt.show()