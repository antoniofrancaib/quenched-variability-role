import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from common_utils import fourier_coefficients

A = 1
B = 1
C = 0  
N = 64
num_trials = 100000


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
import numpy as np
import matplotlib.pyplot as plt

alphas = np.array(alphas)  # Ensure alphas is a numpy array
betas = np.array(betas)    # Ensure betas is a numpy array

R_1 = 2 * np.sqrt(alphas**2 + betas**2)
phi_1 = np.arctan2(betas, alphas)  # Use arctan2 for correct quadrant determination

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(R_1, bins=30, color='green', alpha=0.75)
plt.title('Histogram of $R_1$')
plt.xlabel('$R_1$')
plt.ylabel('Frequency')
plt.axvline(np.sqrt(2 * np.pi / N * (A + C / 2)), color='red', linestyle='dashed', linewidth=2, label='Expected $R_1$')
plt.legend()


# Histogram of phi_1
plt.subplot(1, 2, 2)
plt.hist(phi_1, bins=30, color='purple', alpha=0.75)
plt.title('Histogram of $\\phi_1$')
plt.xlabel('$\\phi_1$ (radians)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
