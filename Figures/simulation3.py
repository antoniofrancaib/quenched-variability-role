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
import numpy as np
import matplotlib.pyplot as plt

alphas = np.array(alphas)  # Ensure alphas is a numpy array
betas = np.array(betas)    # Ensure betas is a numpy array

R_1 = 2 * np.sqrt(alphas**2 + betas**2)
phi_1 = np.arctan2(betas, alphas)  # Use arctan2 for correct quadrant determination

W_1 = 6  # Given constant value for W_1

# Calculate R and delta using the provided formulas
R = np.sqrt(W_1**2 + 2 * W_1 * R_1 * np.cos(phi_1) + R_1**2)
delta = np.arctan2(-R_1 * np.sin(phi_1), W_1 + R_1 * np.cos(phi_1))

# Plotting the results
plt.figure(figsize=(12, 6))

# Histogram of R
plt.subplot(1, 2, 1)
plt.hist(R, bins=30, color='blue', alpha=0.75)
plt.title('Histogram of $R$')
plt.xlabel('$R$')
plt.ylabel('Frequency')
plt.axvline(W_1, linestyle='dashed', linewidth=2, label='$W_1$')


# Histogram of delta
plt.subplot(1, 2, 2)
plt.hist(delta, bins=30, color='green', alpha=0.75)
plt.title('Histogram of $\\delta$')
plt.xlabel('$\\delta$ (radians)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()