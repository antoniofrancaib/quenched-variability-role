import numpy as np
import random
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

np.random.seed(42)
random.seed(42)

L = np.pi
N = 256

w0 = -0.25
I_0 = 1/8

r0= r_02(w0, I_0)
uniform_instability = w0 > 1/derivative_nonlinearity(w0*r0 + I_0)
print(f'Uniform Instability: {uniform_instability}')

amplitudes = []
w1_values = np.linspace(1, 12, 100)
delta_W = 0 

for w1 in w1_values:
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) + delta_W * np.random.randn()

    ring = Ring(L, N, W, I_0, np.cos)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)



critical_w1 = 2/derivative_nonlinearity(w0*r0 + I_0)
print(f'Critical w1: {critical_w1}')


plt.figure(figsize=(10, 6))  
plt.plot(w1_values, amplitudes, 'ro', markersize=2)  
plt.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical $W_1$: {critical_w1:.2f}')  
plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title('Bifurcation Diagram') 
plt.grid(True)  
plt.legend() 
plt.show()  


