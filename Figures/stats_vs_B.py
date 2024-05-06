import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

from common_utils import mean_coefficient_products
import numpy as np
import matplotlib.pyplot as plt

A = 1
N = 64
C = 1  

B_values = np.linspace(0, 1, 10)
alpha_1_sq_numerical = []
beta_1_sq_numerical = []
alpha_2_sq_numerical = []
mean_alpha_1_2_numerical = []
mean_alpha_1_3_numerical = []

num_trials = 10000
for B in B_values:
    alpha_1_sq, beta_1_sq = mean_coefficient_products(1, 0, N, num_trials, A, B, C)
    alpha_2_sq, _ = mean_coefficient_products(2, 0, N, num_trials, A, B, C)
    alpha_1_sq_numerical.append(alpha_1_sq)
    beta_1_sq_numerical.append(beta_1_sq)
    alpha_2_sq_numerical.append(alpha_2_sq)

    mean_product, _ = mean_coefficient_products(1, 1, N, num_trials, A, B, C)
    mean_alpha_1_2_numerical.append(mean_product)
    mean_product, _ = mean_coefficient_products(1, 2, N, num_trials, A, B, C)
    mean_alpha_1_3_numerical.append(mean_product)

alpha_1_sq_theo = (1 / (2 * N)) * (A + (3 * C) / 4) * np.ones_like(B_values)
beta_1_sq_theo = (1 / (2 * N)) * (A + C/ 4) * np.ones_like(B_values)
alpha_2_sq_theo = (1 / (2 * N)) * (A + C / 2) * np.ones_like(B_values)
analytical_mean_product_1_2 = B_values / (4 * (N-1)) 
analytical_mean_product_1_3 = [C / (8 * N)] * np.ones_like(B_values)


plt.figure(figsize=(10, 5))
plt.plot(B_values, alpha_1_sq_theo, label=r'$\langle \alpha_1^2 \rangle$', linestyle='-', color='r')
plt.scatter(B_values, alpha_1_sq_numerical, color='r', marker='o')
plt.plot(B_values, alpha_2_sq_theo, label=r'$\langle \alpha_2^2 \rangle$', linestyle='-', color='g')
plt.scatter(B_values, alpha_2_sq_numerical, color='g', marker='o')
plt.plot(B_values, beta_1_sq_theo, label=r'$\langle \beta_1^2 \rangle$', linestyle='-', color='k')
plt.scatter(B_values, beta_1_sq_numerical, color='k', marker='o')
plt.axhline(y=C / (8 * N), label=r'$\langle \alpha_1 \alpha_3 \rangle$', linestyle='-', color='y')
plt.scatter(B_values, mean_alpha_1_3_numerical, color='y', marker='o')
plt.plot(B_values, analytical_mean_product_1_2, label=r'$\langle \alpha_1 \alpha_2 \rangle$', linestyle='-', color='b')
plt.scatter(B_values, mean_alpha_1_2_numerical, color='b', marker='o')
plt.xlabel('B')
plt.ylabel('Mean Product of Fourier Coefficients')
plt.title('Comparison of Numerical and Analytical Mean Product of Fourier Coefficients')
plt.legend()
plt.grid(True)
plt.show()
