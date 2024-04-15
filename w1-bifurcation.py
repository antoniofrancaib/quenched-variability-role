import numpy as np
import random
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

L = np.pi
N = 256

w0 = -1
I_0 = 0.9

critical_w1 = 2/derivative_nonlinearity(w0*r_02(w0, I_0) + I_0)

amplitudes = []
w1_values = np.linspace(1, 12, 100)

epsilon = 0.1
r_0 = r_02(w0, I_0)
perturbation = lambda theta: r_0 + epsilon * r_0 + epsilon * r_0 * np.cos(theta)

for w1 in w1_values:
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 

    ring = Ring(L, N, W, I_0, perturbation)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)

plt.figure(figsize=(10, 6))  
plt.plot(w1_values, amplitudes, 'ro', markersize=2)  
plt.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical W1: {critical_w1:.2f}')  
plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title('Bifurcation Diagram') 
plt.grid(True)  
plt.legend() 
plt.show()  