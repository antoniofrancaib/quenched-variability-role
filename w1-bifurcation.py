import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from common_utils import Ring, derivative_nonlinearity, r_01, r_02, r_03, fixed_point_solver

np.random.seed(42)
random.seed(42)


L = np.pi
N = 256

w0 = -1
I_0 = 1/8

r0= fixed_point_solver(w0, I_0, initial_guess = r_02(w0, I_0))
print(f'Numerical Solution r0 = {r0}')
print(f'Theoretical Solution r0 = {r_02(w0, I_0)}')

equation1 = lambda w0: w0*derivative_nonlinearity(w0 * r_01(w0, I_0) + I_0) - 1 
equation2 = lambda w0: w0*derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0) - 1 
equation3 = lambda w0: w0*derivative_nonlinearity(w0 * r_03(w0, I_0) + I_0) - 1 

initial_guess = 0.5 
w0_solution1 = fsolve(equation1, initial_guess)[0] # there is something with w0[np.argmax(equation1(w0))], find it
w0_solution2 = fsolve(equation2, initial_guess)[0] # this is equal to 1/4*I0
w0_solution3 = fsolve(equation3, initial_guess)[0]

critical = 2/derivative_nonlinearity(w0*r0 + I_0)
critical_w1 = 2/derivative_nonlinearity(w0_solution1*r_01(w0_solution1, I_0) + I_0)
critical_w2 = 2/derivative_nonlinearity(w0_solution2*r_02(w0_solution2, I_0) + I_0)
critical_w3 = 2/derivative_nonlinearity(w0_solution3*r_03(w0_solution3, I_0) + I_0)

amplitudes = []
w1_values = np.linspace(1, 12, 100)
delta_W = 0 

for w1 in w1_values:
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) + delta_W * np.random.randn()

    ring = Ring(L, N, W, I_0, np.cos)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)

plt.figure(figsize=(10, 6))  
plt.plot(w1_values, amplitudes, 'ro', markersize=2)  
plt.axvline(x=critical, color='b', linestyle='--', label=f'Critical: {critical:.2f}')  
plt.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical $W_1$: {critical_w1:.2f}')  
plt.axvline(x=critical_w2, color='b', linestyle='--', label=f'Critical $W_2$: {critical_w2:.2f}')  
plt.axvline(x=critical_w3, color='b', linestyle='--', label=f'Critical $W_3$: {critical_w3:.2f}')  
plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title('Bifurcation Diagram') 
plt.grid(True)  
plt.legend() 
plt.show()  