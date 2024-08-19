import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from src.utils.common_utils import r_02

def integrand(x, gamma):
    return (np.arctan(x)**2) / (np.pi * gamma * (1 + (x/gamma)**2))

def expected_arctan_squared(gamma):
    result, _ = quad(integrand, -np.inf, np.inf, args=(gamma,))
    return result

A = 1
B = 0
N = 64
w0 = -10
I_0 = 0.9

C_values = np.linspace(0, 100, 1000)  
R_1 = np.sqrt((np.pi * (2*A+C_values) / (4*N)))

variances = []

for C in C_values:
    r_0 = r_02(w0, I_0)
    alpha_sq = (1 / (2 * N)) * (A + (3 * C) / 4) 
    beta_sq = (1 / (2 * N)) * (A + C / 4) 
    gamma = np.sqrt(beta_sq / alpha_sq)
    variance = expected_arctan_squared(gamma)
    variances.append(variance)

plt.figure(figsize=(10, 6))
plt.plot(C_values, variances, label='Variance of $\psi_1$')
plt.plot(C_values, R_1, label='$< R_1 >$')
plt.xlabel('C')
plt.ylabel('Variance')
plt.title('Variance of $\psi_1$ as a Function of C')
plt.legend()
plt.grid(True)
plt.show()
