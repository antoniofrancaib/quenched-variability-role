import numpy as np
import random
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

L = np.pi
N = 256

w0 = -30
I_0 = 2

A = 0.1
B = 0.0 
C = 0.0

R = np.sqrt((2*np.pi/N)*(A+(C/2)))

r_0 = r_02(w0, I_0)
critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R
critical_w2 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) 
print(critical_w1)

amplitudes = []
delta = 0.5
w1_values = np.linspace(critical_w1 -  delta, critical_w1 + delta, 10)

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
    
delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()

for w1 in w1_values:
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) + delta_W(delta_theta)

    ring = Ring(L, N, W, I_0, perturbation)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)

plt.figure(figsize=(10, 6))  
plt.plot(w1_values, amplitudes, 'ro', markersize=2)  
plt.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical W1: {critical_w1:.2f}')  
plt.axvline(x=critical_w2, color='r', linestyle='--', label=f'Critical W1 without quenched variability: {critical_w2:.2f}')  
plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title(f'Bifurcation Diagram for $W_0=${w0} and $I_0=${I_0}') 
plt.grid(True)  
plt.legend() 
plt.show()  