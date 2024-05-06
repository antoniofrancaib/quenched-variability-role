import numpy as np
import matplotlib.pyplot as plt

def V(theta, A, B, C):
    return A + B * np.cos(theta) + C * np.cos(theta)**2

def generate_noise(N):
    return np.random.randn(N)

def delta_W(theta, A, B, C, noise):
    return np.sqrt(V(theta, A, B, C)) * noise


def fourier_coefficients(j, theta_values, delta_W_values):
    cos_terms = np.cos(j * theta_values)
    sin_terms = np.sin(j * theta_values)
    alpha_j = np.trapz(delta_W_values * cos_terms, theta_values) / (2 * np.pi)
    beta_j = np.trapz(delta_W_values * sin_terms, theta_values) / (2 * np.pi)
    return alpha_j, beta_j


M = 60  
s = 0.1  
kappa = s*M

A, B, C = 1, 0, 1
num_trials = 10000  # Number of trials for the numerical calculation
N_values = np.linspace(32, 256, 20, dtype=int)  # N values

analytical_amplitudes = np.sqrt(2 * np.pi / N_values * (A + C / 2)) * np.sqrt(s*M)/ kappa

mean_amplitudes = []  
mean_phases = []  

for N in N_values:
    amplitudes = []  # List to store amplitudes for each trial
    phases = []  # List to store phases for each trial
    theta_values = np.linspace(-np.pi, np.pi, N)  # Generate theta values for current N
    for trial in range(num_trials):
        noise = generate_noise(N)  # Generate noise for current trial
        delta_W_values = delta_W(theta_values, A, B, C, noise)  # Calculate delta_W for current trial
        alpha_j, beta_j = fourier_coefficients(1, theta_values, delta_W_values)  # Get coefficients for j=1
        amplitude = np.sqrt(alpha_j**2 + beta_j**2)  # Calculate amplitude
        phase = np.arctan2(beta_j, alpha_j)  # Calculate phase
        amplitudes.append(amplitude)  # Store amplitude for current trial
        phases.append(phase)  # Store phase for current trial
    mean_amplitude = np.mean(amplitudes)  # Calculate the mean amplitude over all trials
    mean_phase = np.mean(phases)  # Calculate the mean phase over all trials
    mean_amplitudes.append(mean_amplitude)  # Store mean amplitude for current N
    mean_phases.append(mean_phase)  # Store mean phase for current N

plt.figure(figsize=(10, 5))
plt.plot(N_values, mean_amplitudes, marker='o', linestyle='-', color='blue', label='Numerical Mean Amplitude')
plt.plot(N_values, analytical_amplitudes, marker='x', linestyle='--', color='red', label='Analytical Mean Amplitude')
#plt.twinx()  
plt.plot(N_values, mean_phases, marker='s', linestyle='-', color='green', label='Numerical Mean Phase')
plt.ylabel('Mean Phase')
plt.title('Mean Amplitude and Phase as a Function of N')
plt.legend(loc='upper right')
plt.grid(True)

# Original y-axis legend
plt.legend(loc='upper left')
plt.xlabel('N')
plt.ylabel('Mean Amplitude')
plt.show()
