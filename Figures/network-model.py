import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common_utils import nonlinearity

np.random.seed(42)
random.seed(42)

class Neuron:
    def __init__(self, neuron_id, place_field_phase, active=False, activity=0.0):
        self.id = neuron_id
        self.place_field_phase = place_field_phase
        self.active = active
        self.activity = activity

class Environment:
    def __init__(self, k, L, N, M, s, W, initial_activity_function):
        self.id = k
        self.theta = self.initialize_theta(L, N) # spatial plate
        self.neurons = self.initialize_neurons(N, M, s, initial_activity_function)
        self.weight_matrix = self.calculate_weights_matrix(W, L) 
        self.dynamics = self.simulate_dynamics(t_span=(0, 10), t_steps=300)

    def initialize_theta(self, L, N):
        theta_span = (-L, L)
        theta = np.linspace(theta_span[0], theta_span[1], N)
        return theta 
    
    def initialize_neurons(self, N, M, s, initial_activity_function):
        active_per_position = int(s * M)
        neuron_ids = list(range(N * M))
        random.shuffle(neuron_ids)

        temp_neurons = []

        for i in range(N):
            for j in range(M):
                neuron_id = neuron_ids.pop()
                is_active = j < active_per_position
                if is_active:  
                    place_field_phase = self.theta[i]
                    activity_value = initial_activity_function(place_field_phase)
                    temp_neurons.append(Neuron(neuron_id=neuron_id, place_field_phase=place_field_phase, active=True, activity=activity_value))

        return sorted(temp_neurons, key=lambda n: (n.place_field_phase, n.id))


    def calculate_weights_matrix(self, W, L):
        num_neurons = len(self.neurons)
        weights_matrix = np.zeros((num_neurons, num_neurons))

        rho = (L/(num_neurons-1)) * np.concatenate(([0.5], np.ones(num_neurons - 2), [0.5]))

        for i in range(num_neurons):
            for j in range(num_neurons):
                if i != j:
                    delta_theta = self.neurons[i].place_field_phase - self.neurons[j].place_field_phase
                    weights_matrix[i, j] = W(delta_theta) * rho[j]

        return weights_matrix  


    def simulate_dynamics(self, t_span=(0, 10), t_steps=300):
            R0 = np.array([neuron.activity for neuron in self.neurons])

            def dRdt(t, R):
                return -R + nonlinearity(self.weight_matrix @ R)

            t_eval = np.linspace(*t_span, t_steps)
            dynamics = solve_ivp(dRdt, t_span, R0, t_eval=t_eval, method='RK45')

            return dynamics 
    
    def plot_dynamics(self):
            # solve conflict here! neurons with same place fields, average them or just plot them being close to each other?

            theta_span = (self.neurons[0].place_field_phase, self.neurons[-1].place_field_phase)
            theta = np.linspace(theta_span[0], theta_span[1], len(self.neurons))

            T, Y = np.meshgrid(self.dynamics.t, theta)

            plt.figure(figsize=(10, 6))
            c = plt.pcolormesh(T, Y, self.dynamics.y, shading='auto', cmap='viridis')
            plt.colorbar(c, label='Activity Level')
            plt.xlabel('Time')
            plt.ylabel('Neuron Phase')
            plt.title('Neuron Activity Over Time by Phase')
            plt.show()

    def plot_final_state(self):
            # solve conflict here! neurons with same place fields, average them or just plot them being close to each other?

            theta_span = (self.neurons[0].place_field_phase, self.neurons[-1].place_field_phase)
            theta = np.linspace(theta_span[0], theta_span[1], len(self.neurons))

            plt.figure(figsize=(10, 6))  # Set the figure size as desired
            plt.plot(theta, self.dynamics.y[:, -1], label='Neuron Activity at Final Time Step')
            plt.xlabel('Neuron Phase')
            plt.ylabel('Activity Level')
            plt.title('Neuron Activity by Phase at Final Time Step')
            plt.legend()
            plt.show()



def plot_bump_width_vs_W1(W1_values, L, N, M, s, threshold=0.01):
    bump_widths = []
    for W_1 in W1_values:
        W = lambda delta_theta: W0 + W_1 * np.cos(delta_theta) + delta_W * np.random.randn()
        env = Environment(k=1, L=L, N=N, M=M, s=s, W=W, initial_activity_function=cos)
        bump_width = env.calculate_bump_width(threshold)
        bump_widths.append(bump_width)

    plt.figure(figsize=(10, 6))
    plt.plot(W1_values, bump_widths, marker='o')
    plt.xlabel('W_1')
    plt.ylabel('Bump Amplitude')
    plt.title('Bump Width vs. W_1')
    plt.grid(False)
    plt.show()


L = np.pi
N = 256
M = 60  
s = 0.1  
P, D = 0.3, 0.3

W0 = -0.25
W_max = 40
kappa = s*M
eta = 2

a_eta = lambda eta: (2 * P * D / (P + D)) * (1 - P - D)**eta

W_1 = 12 #W_max * ((s*M) / kappa) * a_eta(eta)

delta_W = 0 # W_max * ((s*M) / kappa)**0.5 * V_eta**0.25
W = lambda delta_theta: W0 + W_1 * np.cos(delta_theta) + delta_W * np.random.randn()

