import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from src.utils.common_utils import derivative_nonlinearity, r_02, Ring 

data_folder = os.path.join(project_root, 'data')

L = np.pi
T = {'t_span': (0, 5000), 't_steps': 5000}
B = 0
C = 0
N = 64
w0 = -10
I_0 = 0.9
r_0 = r_02(w0, I_0)

num_simulations = 1 
A_values = np.linspace(0, 10, 10)

def run_simulation(A, w1):
    r_0 = r_02(w0, I_0)
    V = lambda theta: A + B * np.cos(theta) + C * np.cos(theta)**2
    delta_W = lambda theta: np.sqrt(V(theta)) * np.random.randn()
    W = lambda delta_theta: w0 + w1 * np.cos(delta_theta)
    
    ring = Ring(L, T, N, W, delta_W, I_0, lambda theta: r_0 + 0.005 *np.cos(theta), use_quenched_variability=True)
    return ring.calculate_bump_amplitude()

def append_to_csv(results, filename='ring_simulation_results.csv'):
    filepath = os.path.join(data_folder, filename)
    print(f"Saving to file: {filepath}")  # Debug print
    df = pd.DataFrame(results)
    
    if os.path.exists(filepath):
        # If file exists, read it and get the last simulation number
        existing_df = pd.read_csv(filepath)
        last_sim_num = existing_df['simulation_number'].max()
        
        # Update simulation numbers for new results
        df['simulation_number'] += last_sim_num + 1
        
        # Check if 'N' column exists in the existing file
        if 'N' not in existing_df.columns:
            # If 'N' column doesn't exist, add it to the existing file with a default value
            existing_df['N'] = N
            existing_df.to_csv(filepath, index=False)
            
        # Append without writing the header
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        # If file doesn't exist, create it with a header
        df.to_csv(filepath, index=False)

def main():
    all_results = []

    for A in A_values:
        R = np.sqrt((2*np.pi/(N))*(A+(C/2)))
        critical_w1 = 2 / derivative_nonlinearity(w0 * r_0 + I_0) - R
        
        relative_w1_values = [-0.2, -0.1, 0, 0.1, 0.2]
        
        for relative_w1 in relative_w1_values:
            w1 = critical_w1 + relative_w1
            for sim in range(num_simulations):
                amplitude = run_simulation(A, w1)
                all_results.append({
                    'A': A,
                    'w1': w1,
                    'relative_w1': relative_w1,
                    'critical_w1': critical_w1,
                    'amplitude': amplitude,
                    'simulation_number': sim,
                    'N': N
                })

    # Append results to CSV
    append_to_csv(all_results)

    # Read the updated CSV file
    filepath = os.path.join(data_folder, 'ring_simulation_results.csv')
    df = pd.read_csv(filepath)

    print(df.tail())
    print(f"Total number of simulations: {len(df)}")

    critical_w1_summary = df.groupby(['A', 'N'])['critical_w1'].first().reset_index()
    print("\nA values and corresponding critical w1 values:")
    print(critical_w1_summary)

if __name__ == "__main__":
    main()