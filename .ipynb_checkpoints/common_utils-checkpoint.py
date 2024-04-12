import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def nonlinearity(x):
    phi_x = np.zeros_like(x)
    phi_x[(x >= 0) & (x <= 1)] = x[(x >= 0) & (x <= 1)] ** 2
    phi_x[x > 1] = 2 * np.sqrt(x[x > 1]- (3 / 4)) 
    
    return phi_x


def derivative_nonlinearity(x):
    phi_x_prime = np.zeros_like(x)
    phi_x_prime[(x >= 0) & (x <= 1)] = 2 * x[(x >= 0) & (x <= 1)]    
    phi_x_prime[x > 1] = 1 / np.sqrt(x[x > 1] - (3/4))
    
    return phi_x_prime

def dr_dt(t, r, w0, I0):
    return -r + nonlinearity(w0 * r + I0)
    
# FIXED POINT SOLUTIONS for nonlinearity 
r_01 = lambda w0, I_0: (1 - 2 * w0 * I_0 + np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2) # UNSTABLE
r_02 = lambda w0, I_0: (1 - 2 * w0 * I_0 - np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2) # STABLE 
r_03 = lambda w0, I_0: 2*w0 + 2*np.sqrt(w0**2 + I_0 - (3/4)) # STABLE 
r_04 = lambda w0, I_0: 2*w0 - 2*np.sqrt(w0**2 + I_0 - (3/4)) # this one is introduced by squaring **2

def fixed_point_solver(w0, I0, initial_guess = 0.1): # NUMERICAL APPROACH

    equation = lambda r: r - nonlinearity(w0 * r + I0)

    r0_intersection = fsolve(equation, initial_guess)
    
    return r0_intersection[0]


class Ring:
    def __init__(self, L, N, W, external_input, initial_activity_function):
        self.theta = np.linspace(-L, L, N) 
        self.weight_matrix = self.calculate_weights_matrix(L, N, W) 
        self.dynamics = self.simulate_dynamics(external_input, initial_activity_function)

    def calculate_weights_matrix(self, L, N, W):
        weights_matrix = np.zeros((N, N))

        rho = (1/(N-1)) * np.concatenate(([0.5], np.ones(N - 2), [0.5])) # if L != pi, then ((2*L)/((N-1)*2*np.pi))

        for i in range(N):
            for j in range(N):
                    delta_theta = self.theta[i] - self.theta[j]
                    weights_matrix[i, j] = W(delta_theta) * rho[j] 

        return weights_matrix  


    def simulate_dynamics(self, external_input, initial_activity_function, t_span=(0, 30), t_steps=400):
            initial_profile = initial_activity_function(self.theta)

            def dRdt(t, R):
                return -R + nonlinearity((self.weight_matrix @ R)+external_input)  
            
            t_eval = np.linspace(*t_span, t_steps)

            dynamics = solve_ivp(dRdt, t_span, initial_profile, t_eval=t_eval, method='RK45')

            return dynamics 
    
    def plot_dynamics(self):
            T, Y = np.meshgrid(self.dynamics.t, self.theta)

            plt.figure(figsize=(10, 6))
            c = plt.pcolormesh(T, Y, self.dynamics.y, shading='auto', cmap='viridis')
            plt.colorbar(c, label='Activity Level')
            plt.xlabel('Time')
            plt.ylabel('Neuron Phase')
            plt.title('Neuron Activity Over Time by Phase')
            plt.show()

    def plot_final_state(self):
            plt.figure(figsize=(10, 6))  
            plt.plot(self.theta, self.dynamics.y[:, -1], label='Neuron Activity at Final Time Step')
            plt.xlabel('Neuron Phase')
            plt.ylabel('Activity Level')
            plt.title('Neuron Activity by Phase at Final Time Step')
            plt.legend()
            plt.show()
    
    def calculate_bump_amplitude(self):
        return np.max(self.dynamics.y[:, -1]) 