import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, cauchy
from common_utils import fourier_coefficients

A = 1
B = 0
C = 0
N = 64
num_trials = 100000

V = lambda theta, A, B, C: A + B * np.cos(theta) + C * np.cos(theta)**2
delta_W = lambda theta, A, B, C: np.sqrt(np.maximum(V(theta, A, B, C), 0)) * np.random.randn(len(theta))

alpha_variance = 0.5 / N * (A + 3 * C / 4)
beta_variance = 0.5 / N * (A + C / 4)
scale_parameter = np.sqrt(2 * (alpha_variance + beta_variance)) 
gamma = np.sqrt(beta_variance / alpha_variance)
print(gamma)

alphas = []
betas = []
theta_values = np.linspace(-np.pi, np.pi, N)
for _ in range(num_trials):
    delta_W_values = delta_W(theta_values, A, B, C)
    alpha_j, beta_j = fourier_coefficients(2, theta_values, delta_W_values)
    alphas.append(alpha_j)
    betas.append(beta_j)

alphas = np.array(alphas)
betas = np.array(betas)

R_j = 2 * np.sqrt(alphas**2 + betas**2)
phi_j = np.arctan(betas / alphas)

theta_range = np.linspace(-np.pi/2, np.pi/2, 400)
pdf_phi_j = cauchy.pdf(np.tan(theta_range), loc=0, scale=gamma) * (1 + np.tan(theta_range)**2)  

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(R_j, bins=30, color='green', alpha=0.75, density=True, label='Experimental $R_j$')
r = np.linspace(0, np.max(R_j), 200)
plt.plot(r, rayleigh.pdf(r, scale=scale_parameter), 'r-', lw=2, label=f'$\sim$ Rayleigh({scale_parameter:.2f})')
plt.title('Histogram of $R_j$ with Rayleigh Fit')
plt.xlabel('$R_j$')
plt.ylabel('Probability')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(phi_j, bins=30, color='purple', alpha=0.75, density=True, label='Experimental $\\psi_j$')
plt.plot(theta_range, pdf_phi_j, 'r-', lw=2, label='Theoretical $\\psi_j$')
plt.title('Histogram of $\\psi_j$ with Theoretical Fit')
plt.xlabel('$\\psi_j$ (radians)')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()
