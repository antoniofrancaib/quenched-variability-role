import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from common_utils import derivative_nonlinearity, r_02

B = 0
C = 1 
N = 64
w0 = -10
I_0 = 0.9

A_values = np.linspace(0, 10, 100)  

critical_w1_values = []

for A in A_values:
    r_0 = r_02(w0, I_0)
    alpha_sq = (1 / (2 * N)) * (A + (3 * C) / 4) 
    beta_sq = (1 / (2 * N)) * (A + C / 4) 
    sigma = np.sqrt(alpha_sq + beta_sq)
    R_1 = sigma * np.sqrt(np.pi / 2)
    critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R_1
    critical_w1_values.append(critical_w1)

plt.figure(figsize=(10, 6))
plt.plot(A_values, critical_w1_values, label='Critical $w_1$ vs. $A$')
plt.xlabel('$A$')
plt.ylabel('Critical $w_1$')
plt.title('Critical $w_1$ as a Function of $A$')
plt.grid(True)
plt.legend()
plt.show()
