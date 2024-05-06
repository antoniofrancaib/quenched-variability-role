import numpy as np
import matplotlib.pyplot as plt

A = 1
B = 0
C = 0  
N = 64

# Constants
R1 = np.sqrt(2 * np.pi / N * (A + C / 2))
W1 = 1

psi_1 = np.linspace(-np.pi, np.pi, 400)

tan_delta = R1 * np.sin(psi_1) / (W1 + R1 * np.cos(psi_1))
delta = np.arctan(tan_delta)  # Use arctan to get the angle in radians

# Plot delta as a function of psi_1
plt.figure(figsize=(10, 5))
plt.plot(psi_1, delta, label=r'$\delta(\psi_1)$')
plt.title('Plot of $\\delta$ as a Function of $\\psi_1$')
plt.xlabel('$\\psi_1$ (radians)')
plt.ylabel('$\\delta$ (radians)')
plt.grid(True)
plt.legend()
plt.show()

