import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from common_utils import fourier_coefficients
A = 1
B = 0
C = 1  
N = 64
num_trials = 10000


V = lambda theta, A, B, C: A + B * np.cos(theta) + C * np.cos(theta)**2
delta_W = lambda theta, A, B, C: np.sqrt(np.maximum(V(theta, A, B, C), 0)) * np.random.randn(len(theta))

alpha_variance = 0.5 / N * (A + 3 * C / 4)
beta_variance = 0.5 / N * (A + C / 4)

alphas = []
betas = []
theta_values = np.linspace(-np.pi, np.pi, N)

for _ in range(num_trials):
    delta_W_values = delta_W(theta_values, A, B, C)
    alpha_j, beta_j = fourier_coefficients(1, theta_values, delta_W_values)
    alphas.append(alpha_j)
    betas.append(beta_j)

x_range = np.linspace(min(alphas + betas), max(alphas + betas), 300)
norm_alpha = norm.pdf(x_range, 0, np.sqrt(alpha_variance))
norm_beta = norm.pdf(x_range, 0, np.sqrt(beta_variance))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(alphas, bins=30, alpha=0.75, color='b', density=True)
plt.plot(x_range, norm_alpha, 'k--', label=f'N(0, {alpha_variance:.4f})')
plt.title('Histogram of $\\alpha_1$')
plt.xlabel('$\\alpha_1$')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(betas, bins=30, alpha=0.75, color='r', density=True)
plt.plot(x_range, norm_beta, 'k--', label=f'N(0, {beta_variance:.4f})')
plt.title('Histogram of $\\beta_1$')
plt.xlabel('$\\beta_1$')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
