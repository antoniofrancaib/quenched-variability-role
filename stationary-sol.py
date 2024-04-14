import numpy as np
import matplotlib.pyplot as plt
from common_utils import nonlinearity, r_01, r_02, r_03, r_04

w0 = 5
I0 = 0.5

response_functions = [
    (r_01, 'red', '01'),
    (r_02, 'orange', '02'),
    (r_03, 'purple', '03'),
    (r_04, 'green', '04')  
]

r0_values = np.linspace(-2.5, 15, 1000)
phi_values = nonlinearity(w0 * r0_values + I0)
plt.fill_between(r0_values, -2.5, 15, where=(w0 * r0_values + I0 >= 0) & (w0 * r0_values + I0 <= 1), color='lightblue', alpha=0.3, label='$0 \leq (w_0 \cdot r_0 + I_0) \leq 1$')
plt.fill_between(r0_values, -2.5, 15, where=w0 * r0_values + I0 > 1, color='lightgray', alpha=0.3, label='$(w_0 \cdot r_0 + I_0) > 1$')
plt.plot(r0_values, phi_values, label='$\phi(w_0 \cdot r_0 + I_0)$', color='blue')
plt.plot(r0_values, r0_values, label='$r_0$', linestyle='--', color='red')

tolerance = 1e-6
for func, color, label in response_functions:
    r_val = func(w0, I0) 
    if np.isclose(r_val, nonlinearity(w0 * r_val + I0), atol=tolerance):
        plt.scatter([r_val], [nonlinearity(w0 * r_val + I0)], color=color, label=f'$r_{{{label}}}$: {r_val:.2f}', zorder=5)

plt.gca().set_facecolor('white')
plt.gcf().set_facecolor('white')
plt.xlabel('$r_0$', color='black')
plt.ylabel('Value', color='black')
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.legend()
plt.grid(color='gray')

plt.show()
