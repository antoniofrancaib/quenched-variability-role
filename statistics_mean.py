import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

class Neuron:
    def __init__(self, neuron_id, place_field_phase, active=False):
        self.id = neuron_id
        self.place_field_phase = place_field_phase
        self.active = active  

    def activate(self):
        self.active = True


class Environment:
    def __init__(self, k, L, N, M, s, W, P, D):
        self.id = k
        self.neurons = []
        self.M = M
        self.s = s
        self.initialize_neurons(L, N, M, s)

    def initialize_neurons(self, L, N, M, s): # determine active neurons for this environment
        theta_span = (-L, L)
        theta = np.linspace(theta_span[0], theta_span[1], N)

        active_per_position = int(s * M)

        neuron_ids = list(range(N * M))
        random.shuffle(neuron_ids)

        for i in range(N):
            for j in range(M):
                neuron_id = neuron_ids.pop()
                is_active = j < active_per_position
                self.neurons.append(Neuron(neuron_id=neuron_id, place_field_phase=theta[i], active=is_active))

    def order_active_neurons_by_phase(self):
        active_neurons = [neuron for neuron in self.neurons if neuron.active]
        ordered_neurons = sorted(active_neurons, key=lambda n: (n.place_field_phase, n.id))
        return ordered_neurons

    def get_connectivity_matrix(self):
        f_p = lambda delta_theta: 1 + np.cos(delta_theta)
        f_D = lambda delta_theta: 1 - np.cos(delta_theta)

        for neuron_i in self.neurons:
            for neuron_j in self.neurons: 
                if (neuron_i.id != neuron_j.id) and neuron_j.active and neuron_j.active:
                    W[neuron_i.id, neuron_j.id] += P*(1-W[neuron_i.id, neuron_j.id])*f_p(neuron_i.place_field_phase - neuron_j.place_field_phase) - D*W[neuron_i.id, neuron_j.id]*f_D(neuron_i.place_field_phase - neuron_j.place_field_phase)
        return W
    
    def print_active_neurons(self):
        ordered_active_neurons = self.order_active_neurons_by_phase()
        for neuron in ordered_active_neurons:
            print(f"Neuron ID: {neuron.id}, Phase: {neuron.place_field_phase}, Active: {neuron.active}")


L = np.pi
N = 100  
M = 5  
s = 0.5  
W = np.zeros((N*M, N*M))
P, D = 0.3, 0.3

W = np.zeros((N*M, N*M))

mean_connectivity_values = []

mu_w_theoretical = P / (P + D)

for k in range(1, 21):  
    env = Environment(k=k, L=L, N=N, M=M, s=s, W=W, P=P, D=D)
    W = env.get_connectivity_matrix()  
    mean_value = np.mean(W)
    print(f"Environment {k}, Mean Connectivity: {mean_value}")
    mean_connectivity_values.append(mean_value)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), mean_connectivity_values, marker='o', linestyle='-', color='b', label='Empirical Mean Connectivity')
plt.axhline(y=mu_w_theoretical, color='r', linestyle='-', label=f'Theoretical Value: {mu_w_theoretical}')

plt.xticks(range(1, 21))

plt.title('Mean Connectivity as a Function of Environment Number')
plt.xlabel('Environment Number (k)')
plt.ylabel('Mean Connectivity')
plt.legend()
plt.grid(True)
plt.show()
