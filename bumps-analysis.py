import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the CSV file
df = pd.read_csv('ring_simulation_results.csv')

# Filter the data for N=128
df = df[df['N'] == 64]

amplitude_threshold = 0.5  # Adjust this value as needed

print(f"r_0: {df['critical_w1'].iloc[0] + np.sqrt((2*np.pi/256)*(df['A'].iloc[0]))}")

# First plot: Number of Bumps Formed
bump_counts = df.groupby(['A', 'relative_w1']).agg({
    'amplitude': lambda x: (x > amplitude_threshold).sum(),
    'critical_w1': 'first'  
}).reset_index()
bump_counts.columns = ['A', 'relative_w1', 'bump_count', 'critical_w1']

bump_counts['w1'] = bump_counts['critical_w1'] + bump_counts['relative_w1']

plt.figure(figsize=(12, 8))

scatter = plt.scatter(bump_counts['A'], bump_counts['w1'], 
                      c=bump_counts['bump_count'], s=bump_counts['bump_count']*5,
                      cmap='viridis', alpha=0.7)

plt.colorbar(scatter, label='Number of Bumps Formed')

A_values = bump_counts['A'].unique()
critical_w1_values = bump_counts.groupby('A')['critical_w1'].first()
plt.plot(A_values, critical_w1_values, label='Critical $w_1$', color='red', linewidth=2)

plt.title('Number of Bumps Formed for Different A-w1 Pairs (N=128)')
plt.xlabel('A (Noise Amplitude)')
plt.ylabel('$w_1$')
plt.legend(loc='upper right')

plt.ylim(bump_counts['w1'].min() - 0.1, bump_counts['w1'].max() + 0.1)
plt.xlim(A_values.min() - 0.5, A_values.max() + 0.5)

plt.tight_layout()
plt.show()

# Second plot: Amplitude Statistics
amplitude_stats = df.groupby(['A', 'w1']).agg({
    'amplitude': ['mean', 'std', 'count']
}).reset_index()

amplitude_stats.columns = ['A', 'w1', 'mean_amplitude', 'std_amplitude', 'count']

def calculate_ci(row):
    df = row['count'] - 1
    t_value = stats.t.ppf(0.975, df)
    ci = t_value * (row['std_amplitude'] / np.sqrt(row['count']))
    return pd.Series({'ci_lower': row['mean_amplitude'] - ci, 
                      'ci_upper': row['mean_amplitude'] + ci})

amplitude_stats[['ci_lower', 'ci_upper']] = amplitude_stats.apply(calculate_ci, axis=1)

plt.figure(figsize=(12, 8))

scatter = plt.scatter(amplitude_stats['A'], amplitude_stats['w1'], 
                      c=amplitude_stats['mean_amplitude'], s=50,
                      cmap='viridis', alpha=0.7)

plt.colorbar(scatter, label='Mean Amplitude')

for idx, row in amplitude_stats.iterrows():
    annotation_text = f"Mean: {row['mean_amplitude']:.2f}\nCI: [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
    plt.annotate(annotation_text, (row['A'], row['w1']), 
                 xytext=(5, 5), textcoords='offset points', 
                 fontsize=6, alpha=0.7, 
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.7))

plt.plot(A_values, critical_w1_values, label='Critical $w_1$', color='red', linewidth=2)

plt.title('Amplitude Statistics for Different A-w1 Pairs (N=128)')
plt.xlabel('A (Noise Amplitude)')
plt.ylabel('$w_1$')
plt.legend(loc='upper right')

plt.ylim(amplitude_stats['w1'].min() - 0.1, amplitude_stats['w1'].max() + 0.1)
plt.xlim(A_values.min() - 0.5, A_values.max() + 0.5)

plt.tight_layout()
plt.show()