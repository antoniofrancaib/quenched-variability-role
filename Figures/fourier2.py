import numpy as np
import matplotlib.pyplot as plt

# Define the potential function V
def V(theta, A, B, C):
    return A + B * np.cos(theta) + C * np.cos(theta)**2

# Calculate delta_W
def delta_W(theta, A, B, C):
    return np.sqrt(V(theta, A, B, C)) * np.random.randn(len(theta))

# Function to calculate Fourier coefficients
def fourier_coefficients(j, theta_values, delta_W_values):
    cos_terms = np.cos(j * theta_values)
    sin_terms = np.sin(j * theta_values)
    alpha_j = np.trapz(delta_W_values * cos_terms, theta_values) / (2 * np.pi)
    beta_j = np.trapz(delta_W_values * sin_terms, theta_values) / (2 * np.pi)
    return alpha_j, beta_j

# Parameters
A, B, C = 1, 0.5, 0.3  # Example coefficients for V
N = 64  # Number of theta points
theta_values = np.linspace(0, 2 * np.pi, N)
delta_W_values = delta_W(theta_values, A, B, C)

# Fourier series reconstruction
def fourier_series(theta_values, J):
    fourier_reconstruction = np.zeros_like(theta_values)
    for j in range(1, J+1):
        alpha_j, beta_j = fourier_coefficients(j, theta_values, delta_W_values)
        fourier_reconstruction += alpha_j * np.cos(j * theta_values) + beta_j * np.sin(j * theta_values)
    return fourier_reconstruction

# Maximum mode J
J = 1  # Example, up to the 10th mode

# Calculate the Fourier series up to J-th mode
reconstructed_signal = fourier_series(theta_values, J)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(theta_values, delta_W_values, label='Original $\Delta W$', alpha=0.5)
plt.plot(theta_values, reconstructed_signal, label=f'Fourier Series up to {J} modes', linestyle='--')
plt.title("Fourier Series Approximation of $\Delta W$")
plt.xlabel("$\\theta$")
plt.ylabel("$\Delta W$")
plt.legend()
plt.show()
