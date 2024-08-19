import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from src.utils.common_utils import derivative_nonlinearity, r_02, Ring 
from scipy.stats import sem, t


L = np.pi
T = {'t_span': (0, 5000), 't_steps': 5000}
B = 0
C = 0
N = 64
w0 = -10
I_0 = 0.9
r_0 = r_02(w0, I_0)

num_simulations = 100

A_values = np.linspace(0, 10, 10)
theoretical_critical_w1_values = []
bump_amplitudes = []

def run_simulation(A, w1):
    r_0 = r_02(w0, I_0)
    V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
    delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta)
    
    ring = Ring(L, T, N, W, delta_W, I_0, lambda theta: r_0 + 0.005 *np.cos(theta), use_quenched_variability=True)
    return ring.calculate_bump_amplitude()

def calculate_ci(data):
    mean = np.mean(data)
    se = sem(data)
    ci = t.interval(0.95, len(data)-1, loc=mean, scale=se)
    return max(0, ci[0]), ci[1]  # Ensure lower bound is non-negative

for A in A_values:
    R = np.sqrt((2*np.pi/(N))*(A+(C/2)))
    critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R
    theoretical_critical_w1_values.append(critical_w1)
    
    w1_values = [critical_w1 - 0.2, critical_w1 - 0.1, critical_w1, critical_w1 + 0.1, critical_w1 + 0.2]
    amplitudes = []
    
    for w1 in w1_values:
        simulation_results = [run_simulation(A, w1) for _ in range(num_simulations)]
        mean_amplitude = np.mean(simulation_results)
        confidence_interval = calculate_ci(simulation_results)
        amplitudes.append((w1, mean_amplitude, confidence_interval))
    
    bump_amplitudes.append(amplitudes)

plt.figure(figsize=(12, 8))

plt.plot(A_values, theoretical_critical_w1_values, label='Theoretical Critical $w_1$', color='blue')

for i, (A, amplitudes) in enumerate(zip(A_values, bump_amplitudes)):
    w1_values, mean_amps, _ = zip(*amplitudes)
    plt.scatter([A]*5, w1_values, c=mean_amps, cmap='viridis', s=50, zorder=10)
    for w1, mean_amp, ci in amplitudes:
        if A == 0:
            annotation_text = f'Mean: {mean_amp:.2f}'
        else:
            annotation_text = f'Mean: {mean_amp:.2f}\nCI: [{ci[0]:.2f}, {ci[1]:.2f}]'
        plt.annotate(annotation_text, (A, w1), xytext=(5, 0), textcoords='offset points', fontsize=6)

scatter = plt.scatter([], [], c=[], cmap='viridis')
plt.colorbar(scatter, label='Mean Bump Amplitude')

plt.xlabel('$A$')
plt.ylabel('$w_1$')
plt.title('Theoretical Critical $w_1$ and Mean Bump Amplitudes as a Function of $A$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()