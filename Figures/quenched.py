import numpy as np
import random
import matplotlib.pyplot as plt
from common_utils import Ring, derivative_nonlinearity, r_02

# PARAMETERS
L = np.pi
N = 256
M = 60  
s = 0.1  
P, D = 0.3, 0.3

W_max = 40
kappa = s*M

# STATISTICS for f_p = 1 + cos and f_d = 1 - cos
mu = P / (P + D)
sigma_squared = 2*D**2*P**2 / ((P + D)**2 * (2*D*P + 2*D + 2*P - 1.5*(D + P)**2))

# FUZZY 
f_p = 1
f_d = 1
f_p2 = 1.5
f_d2 = 1.5 
w_sq = mu * (-D * P + 3 * P**2 - 4 * P) / (3 * D**2 + 2 * D * P - 4 * D + 3 * P**2 - 4 * P)
F_avg = -D - P + 1
F_sq_avg = 3 * D**2 / 2 + D * P - 2 * D + 3 * P**2 / 2 - 2 * P + 1


# COEFFICIENTS
A_0 = P**2 + 2*P*mu*(-D - P + 1) - mu**2 + w_sq*(-D - P + 1)**2
B_0 = (-2*D + 2*P)*(P**2 + P*mu*(-2*D - 2*P + 1) - w_sq*(D + P)*(-D - P + 1))/(D + P)
C_0 = (-D + P)**2*(P**2 - 2*P*mu*(D + P) + w_sq*(D + P)**2)/(D + P)**2

a_eta = lambda eta: (2 * P * D / (P + D)) * (1 - s**2*(P + D))**eta


A_eta = lambda eta: A_0 * (F_sq_avg)**eta + mu**2 * ((F_sq_avg)**eta - 1) + \
    3/2 * P**2 * s**2 * (1 - (F_sq_avg)**eta) / (1 - F_sq_avg) + \
    2*P**2 * s**4 * (1 - 3/2 * P - 1/2 * D) * (1/(F_avg - F_sq_avg) * (1 - (F_avg)**eta)/(1 - F_avg) - (1 - (F_sq_avg)**eta)/(1 - F_sq_avg)) + \
    2*mu*P * s**2 * (1 - 3/2 * P - 1/2 * D) * ((F_avg)**eta - (F_sq_avg)**eta) / (F_avg - F_sq_avg)


B_eta = lambda eta: B_0 * (F_sq_avg)**eta + 2*a_eta(0)*mu * ((F_sq_avg)**eta - (F_avg)**(2*eta)) + \
    2*a_eta(0)*P * s**2 * (1 - 3/2 * P - 1/2 * D) * (F_avg**eta - F_sq_avg**eta) / (F_avg - F_sq_avg) - \
    (F_avg**eta - F_sq_avg**(2*eta)) / (1 - F_avg)


C_eta = lambda eta: C_0 * (F_sq_avg)**eta + a_eta(0)**2 * ((F_sq_avg)**eta - (F_avg)**(2*eta))


V_eta = lambda eta, theta: A_eta(eta) + B_eta(eta) * np.cos(theta) + C_eta(eta) * np.cos(theta)**2

w0 = -30
I_0 = 2

r_0 = r_02(w0, I_0)
critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0)

epsilon = 0.005
perturbation = lambda theta: r_0 + epsilon * np.cos(theta)

eta_values = np.linspace(0,50, 5)

amplitudes = []

for eta in eta_values:
    W_1 = W_max * ((s*M) / kappa) * a_eta(eta)

    delta_W = lambda eta, delta_theta: W_max * ((s*M) / kappa)**0.5 * V_eta(eta, delta_theta)**0.25

    W = lambda delta_theta: w0 + W_1 * np.cos(delta_theta) + delta_W(eta, delta_theta) * np.random.randn()

    ring = Ring(L, N, W, I_0, perturbation)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)
"""
plt.figure(figsize=(10, 6))  
plt.plot(eta_values, amplitudes, 'ro', markersize=2)  
plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title(f'Bifurcation Diagram for $W_0=${w0} and $I_0=${I_0}') 
plt.grid(True)  
plt.legend() 
plt.show()  
""" 

r_0 = r_02(w0, I_0)
critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0)

W_1_values = W_max * ((s * M) / kappa) * a_eta(eta_values)

fig, ax1 = plt.subplots()

# Bottom x-axis (W_1 values)
color = 'tab:red'
ax1.set_xlabel('W_1', color=color)
ax1.set_ylabel('Amplitude', color=color)
ax1.plot(W_1_values, amplitudes, 'ro-', markersize=2, color=color)
ax1.tick_params(axis='x', labelcolor=color)

# Add vertical line for critical_w1
ax1.axvline(x=critical_w1, color='b', linestyle='--', label=f'Critical W1: {critical_w1:.2f}')
ax1.legend()

# Upper x-axis (eta values)
ax2 = ax1.twiny()
color = 'tab:blue'
ax2.set_xlabel('eta', color=color)
ax2.tick_params(axis='x', labelcolor=color)

# Align the eta ticks with the W_1 values and label them
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(W_1_values)
ax2.set_xticklabels([f"{eta:.1f}" for eta in eta_values])

fig.tight_layout()
plt.show()