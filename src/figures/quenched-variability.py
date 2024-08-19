import numpy as np
import matplotlib.pyplot as plt

L = np.pi
N = 256

theta = np.linspace(-np.pi, np.pi, N)

w0 = -10 
w1 = 10

A = 1
B = 0
C = 1

V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2

V_non_negative = np.maximum(V(theta), 0) #np.abs(V(theta))
delta_W = lambda theta: np.sqrt(V_non_negative) * np.random.randn(len(theta))

W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) + delta_W(delta_theta)

fig, axs = plt.subplots(1, 3, figsize=(24, 6)) 

axs[0].plot(theta, V(theta), 'r-', markersize=2)
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].set_xlabel('$\\theta$')
axs[0].set_ylabel('$V(\\theta)$')
axs[0].set_title('$V(\\theta)$')
axs[0].grid(True)
axs[0].legend(['V'])

axs[1].plot(theta, delta_W(theta), 'b-', markersize=2)
axs[1].set_xlabel('$\\theta$')
axs[1].set_ylabel('$\Delta W(\\theta)$')
axs[1].set_title('$\Delta W(\\theta)$')
axs[1].grid(True)
axs[1].legend(['$\Delta W$'])

axs[2].plot(theta, W(theta), 'g-', markersize=2)
axs[2].set_xlabel('$\\theta$')
axs[2].set_ylabel('$W(\\theta)$')
axs[2].set_title('$W(\\theta)$')
axs[2].grid(True)
axs[2].legend(['$W$'])

plt.show()