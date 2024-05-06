import numpy as np
import matplotlib.pyplot as plt
from common_utils import nonlinearity, fixed_point_solver, derivative_nonlinearity, r_02


class RingEuler:
    def __init__(self, theta, W, I_0, nonlinearity):
        self.theta = theta
        self.kernel = W 
        self.r = self.simulate_dynamics(I_0, nonlinearity)
    
    def simulate_dynamics(self, I_0, nonlinearity, num_t=200, delta_t=0.1):
        r = np.zeros((len(self.theta), num_t))

        r[:, 0] = np.cos(self.theta)  

        for t in range(1, num_t):
            for i in range(len(self.theta)):
                integral_sum = 0.0
                for j in range(len(self.theta)):
                    integral_sum += self.kernel(self.theta[i] - self.theta[j]) * r[j, t-1]
                integral_sum *= (1/(2*np.pi))

                r[i, t] = r[i, t-1] + delta_t * (-r[i, t-1] + nonlinearity(integral_sum) + I_0)

        return r
    
    def plot_dynamics(self, num_t=200, delta_t=0.1):
        theta_grid, t_grid = np.meshgrid(self.theta, np.arange(0, num_t * delta_t, delta_t))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('r')
        ax.yaxis.pane.set_edgecolor('g')
        ax.zaxis.pane.set_edgecolor('b')
        ax.grid(False)

        surf = ax.plot_surface(theta_grid, t_grid, self.r.T, cmap='viridis', edgecolor='none')

        ax.set_xlabel('Theta')
        ax.set_ylabel('Time')
        ax.set_zlabel('Activity')

        plt.show()


    def plot_final_state(self):
            plt.figure(figsize=(10, 6))  
            plt.plot(self.theta, self.r[:, -1], label='Neuron Activity at Final Time Step')
            plt.xlabel('Neuron Phase')
            plt.ylabel('Activity Level')
            plt.title('Neuron Activity by Phase at Final Time Step')
            plt.legend()
            plt.show()

    def calculate_bump_amplitude(self):
        return np.max(self.r[:, -1]) 
    
num_theta=100
theta = np.linspace(-np.pi, np.pi, num_theta)

W_0 = -1 
I_0 = 0.3

"""W_1=1
W = lambda delta_theta: W_0 + W_1 * np.cos(delta_theta)
ring = RingEuler(theta, W, I_0, nonlinearity)
ring.plot_final_state()
"""
r0= fixed_point_solver(W_0, I_0, initial_guess = r_02(W_0, I_0))
critical = 2/derivative_nonlinearity(W_0*r0 + I_0)

amplitudes = []
w1_values = np.linspace(0, 8, 20)

for W_1 in w1_values:
    W = lambda delta_theta: W_0 + W_1 * np.cos(delta_theta)

    ring = RingEuler(theta, W, I_0, nonlinearity)  
    amplitude = ring.calculate_bump_amplitude()  
    amplitudes.append(amplitude)

plt.figure(figsize=(10, 6))  
plt.plot(w1_values, amplitudes, 'ro', markersize=2)  
plt.axvline(x=critical, color='b', linestyle='--', label=f'Critical: {critical:.2f}')  
plt.xlabel('$W_1$ Values')  
plt.ylabel('Amplitude') 
plt.title('Bifurcation Diagram') 
plt.grid(True)  
plt.legend() 
plt.show()  
