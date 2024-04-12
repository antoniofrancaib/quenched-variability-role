import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import norm
from common_utils import nonlinearity

N_theta = 100  # Number of discretized points on the ring
theta = np.linspace(-np.pi, np.pi, N_theta)  # Discretize the domain
dt = 0.01  # Time step size
T = 10  # Total time of simulation
N_t = int(T / dt)  # Number of time steps
tau = 1.0  # Time constant
I0 = 1.0  # External input

# Initialize state variables
r = np.zeros(N_theta)

w0 = -1
w1 = 6

# Define connectivity as a function
W = lambda delta_theta: w0 + w1 * np.cos(delta_theta) 


# Time stepping
for t in range(N_t):
    # Compute the integral term (using the trapezoidal rule or Simpson's rule)
    integral_term = np.zeros(N_theta)
    for i in range(N_theta):
        delta_theta = theta[i] - theta
        M = W(delta_theta)
        integral_term[i] = simps(M * r, theta) / (2 * np.pi)
    
    # using the Euler method
    dr_dt = -r + nonlinearity(integral_term + I0)
    r = r + dt * dr_dt
    