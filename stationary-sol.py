import numpy as np
import matplotlib.pyplot as plt
from common_utils import fixed_point_solver, nonlinearity, derivative_nonlinearity, r_01, r_02, r_03

# conclusion from numerics: 
# - only one solution (r_02) for w0<1(approx)
# - three solutions (r_01, r_02, r_03) for 1(approx)<w0<1/4I0
# - two solutions (r_01=r_02, r_03) for w0=1/4I0
# - one solution (r_03) for w0>1/4I0

w0 = 5
I0 = 0.5

r_01_val = r_01(w0, I0) 
r_02_val = r_02(w0, I0) 
r_03_val = r_03(w0, I0) 

r0_values = np.linspace(-5, 18, 1000)
phi_values = nonlinearity(w0 * r0_values + I0)

plt.figure(figsize=(10, 6))

plt.fill_between(r0_values, -10, 15, where=(w0 * r0_values + I0 >= 0) & (w0 * r0_values + I0 <= 1), color='lightblue', alpha=0.3, label='$0 \leq (w_0 \cdot r_0 + I_0) \leq 1$')
plt.fill_between(r0_values, -10, 15, where=w0 * r0_values + I0 > 1, color='lightcoral', alpha=0.3, label='$(w_0 \cdot r_0 + I_0) > 1$')

plt.plot(r0_values, phi_values, label='$\phi(w_0 \cdot r_0 + I_0)$', color='blue')
plt.plot(r0_values, r0_values, label='$r_0 = r_0$', linestyle='--', color='red')


tolerance=1e-4
if np.isclose(r_01_val, nonlinearity(w0*r_01_val+I0), atol=tolerance):
    plt.scatter([r_01_val], [nonlinearity(w0*r_01_val+I0)], color='red', label=f'$r_{{01}}$: {r_01_val:.6f}', zorder=5)
    
if np.isclose(r_02_val, nonlinearity(w0*r_02_val+I0), atol=tolerance):
    plt.scatter([r_02_val], [nonlinearity(w0*r_02_val+I0)], color='orange', label=f'$r_{{02}}$: {r_02_val:.6f}', zorder=5)

if np.isclose(r_03_val, nonlinearity(w0*r_03_val+I0), atol=tolerance):
    plt.scatter([r_03_val], [nonlinearity(w0*r_03_val+I0)], color='purple', label=f'$r_{{03}}, \phi(w_0 r_{{03}}+I_0)$: ({r_03_val:.2f}, {nonlinearity(w0*r_03_val+I0):.2f})', zorder=5)

plt.gca().set_facecolor('white')
plt.gcf().set_facecolor('white')
plt.xlabel('$r_0$', color='black')
plt.ylabel('Value', color='black')
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.legend()
plt.grid(color='gray')

plt.show()

