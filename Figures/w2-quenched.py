import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02


L = np.pi
N = 256
T = {'t_span': (0, 10000), 't_steps': 10000}

w0 = -10
I_0 = 0.9

A = 0.1
B = 0.0 
C = 0.0

M = 60  
s = 0.1  
kappa = s*M

R = np.sqrt((2*np.pi/N)*(A+(C/2)))#* (np.sqrt(s*M)/kappa)

r_0 = r_02(w0, I_0)
critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R
wout_critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) 
print(critical_w1)

amplitudes = []
delta = 0.5
w1_values = np.linspace(critical_w1 -  delta, critical_w1 + delta, 30)

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
    
delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()

num_simulations = 20
all_simulation_amplitudes = []


for w1 in w1_values:
    simulation_amplitudes = []
    for _ in range(num_simulations):
        W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) + delta_W(delta_theta)
        ring = Ring(L, T, N, W, I_0, perturbation)  
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
