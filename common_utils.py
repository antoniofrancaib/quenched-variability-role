import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import math 

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
r_01 = lambda w0, I_0: (1 - 2 * w0 * I_0 + np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2) 
r_02 = lambda w0, I_0: (1 - 2 * w0 * I_0 - np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2) 
r_03 = lambda w0, I_0: 2*w0 + 2*np.sqrt(w0**2 + I_0 - (3/4)) 
r_04 = lambda w0, I_0: 2*w0 - 2*np.sqrt(w0**2 + I_0 - (3/4)) 

def fourier_coefficients(j, theta_values, delta_W_values):
    cos_terms = np.cos(j * theta_values)
    sin_terms = np.sin(j * theta_values)
    alpha_j = np.trapz(delta_W_values * cos_terms, theta_values) / (2 * np.pi)
    beta_j = np.trapz(delta_W_values * sin_terms, theta_values) / (2 * np.pi)
    return alpha_j, beta_j

def mean_coefficient_products(j, delta, N=64, num_trials=10000, A=1, B=0.0, C=0):
    V = lambda theta, A, B, C: A + B * np.cos(theta) + C * np.cos(theta)**2
    delta_W = lambda theta, A, B, C: np.sqrt(V(theta, A, B, C)) * np.random.randn(len(theta))

    theta_values = np.linspace(-np.pi, np.pi, N)
    alpha_product_values = np.zeros(num_trials)
    beta_product_values = np.zeros(num_trials)
    
    for trial in range(num_trials):
        delta_W_values = delta_W(theta_values, A, B, C)
        alpha_j, beta_j = fourier_coefficients(j, theta_values, delta_W_values)
        alpha_j_delta, beta_j_delta = fourier_coefficients(j + delta, theta_values, delta_W_values)
        
        if delta == 0:  # Special case for the square of alpha_j and beta_j
            alpha_product_values[trial] = alpha_j ** 2
            beta_product_values[trial] = beta_j ** 2
        else:
            alpha_product_values[trial] = alpha_j * alpha_j_delta
            beta_product_values[trial] = beta_j * beta_j_delta
        
    return np.mean(alpha_product_values), np.mean(beta_product_values)

def fixed_point_solver(w0, I0, initial_guess = 0.1): # NUMERICAL APPROACH

    equation = lambda r: r - nonlinearity(w0 * r + I0)

    r0_intersection = fsolve(equation, initial_guess)
    
    return r0_intersection[0]


def find_critical_w0(r_func, I_0, initial_guess=0.5):
    equation = lambda w0: w0 * derivative_nonlinearity(w0 * r_func(w0, I_0) + I_0) - 1
    
    critical_w0 = fsolve(equation, initial_guess)[0]
    return critical_w0


def apply_mask(w0, I_0, r_func, tolerance=1e-4):
    r_values = r_func(w0, I_0)
    mask = np.isclose(r_values, nonlinearity(w0 * r_values + I_0), atol=tolerance)
    
    r_values_filtered = np.where(mask, r_values, np.nan)
    w0_filtered = np.where(mask, w0, np.nan)  

    return w0_filtered, r_values_filtered

def select_and_solve(w0, r_filtered, I_0, func):
    non_nan_indices = ~np.isnan(r_filtered)
    w0_non_nan = w0[non_nan_indices]
        
    if len(w0_non_nan) == 0:
       return np.array([]), np.array([])  
        
    indices_selected = np.linspace(0, len(w0_non_nan) - 1, min(len(w0_non_nan), 10), dtype=int)
    w0_selected = w0_non_nan[indices_selected]
    r_num = []
    t_span = [0, 100]
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
        
    for w0_val in w0_selected:
        r0 = [func(w0_val, I_0)+0.1] 
        sol = solve_ivp(dr_dt, t_span, r0, args=(w0_val, I_0), t_eval=t_eval, method='RK45')
        r_num.append(sol.y[0, -1])
        
    return w0_selected, r_num

class Ring:
    def __init__(self, L, T, N, W, external_input, initial_activity_function):
        self.theta = np.linspace(-L, L, N) 
        self.weight_matrix = self.calculate_weights_matrix(L, N, W) 
        self.dynamics = self.simulate_dynamics(T, external_input, initial_activity_function)

    def calculate_weights_matrix(self, L, N, W):
        weights_matrix = np.zeros((N, N))

        rho = (1/(N-1)) * np.concatenate(([0.5], np.ones(N - 2), [0.5])) # if L != pi, then ((2*L)/((N-1)*2*np.pi))

        for i in range(N):
            for j in range(N):
                    delta_theta = self.theta[i] - self.theta[j]
                    weights_matrix[i, j] = W(delta_theta) * rho[j] 

        return weights_matrix  


    def simulate_dynamics(self, T, external_input, initial_activity_function):
            t_span = T['t_span']
            t_steps = T['t_steps']

            initial_profile = initial_activity_function(self.theta)

            def dRdt(t, R):
                return -R + nonlinearity((self.weight_matrix @ R)+external_input)  
            
            t_eval = np.linspace(*t_span, t_steps)

            dynamics = solve_ivp(dRdt, t_span, initial_profile, t_eval=t_eval, method='RK45')

            return dynamics 
    
    def plot_dynamics(self, ax):
        T, Y = np.meshgrid(self.dynamics.t, self.theta)
        c = ax.pcolormesh(T, Y, self.dynamics.y, shading='auto', cmap='viridis')
        plt.colorbar(c, ax=ax, label='Activity Level')  # Changed to plt.colorbar to fetch the current figure implicitly
        ax.set_xlabel('Time')
        ax.set_ylabel('Neuron Phase')
        ax.set_title('Neuron Activity Over Time by Phase')

    def plot_state(self, ax, timestep=-1):
        ax.plot(self.theta, self.dynamics.y[:, timestep], label='Neuron Activity at Final Time Step')
        ax.set_xlabel('Neuron Phase')
        ax.set_ylabel('Activity Level')
        ax.set_title(f'Neuron Activity by Phase at Time Step {timestep}')
        ax.legend()

    def plot_timetrace(self, ax, phase=None):

        if phase is None:
            index = len(self.theta) // 2
        else:
            index = np.argmin(np.abs(self.theta - phase))

        ax.plot(self.dynamics.t, self.dynamics.y[index, :], label=f'Activity at Phase {self.theta[index]:.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity Level')
        ax.set_title(f'Time Trace of Neuron Activity at Phase {self.theta[index]:.2f}')
        ax.legend()

    def calculate_bump_amplitude(self):
        return np.max(self.dynamics.y[:, -1]) - np.min(self.dynamics.y[:, -1]) 
    

    