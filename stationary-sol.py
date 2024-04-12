import numpy as np
import matplotlib.pyplot as plt
from common_utils import fixed_point_solver, nonlinearity, derivative_nonlinearity, r_01, r_02, r_03, r_04

# conclusion from numerics: 
# - only one solution (r_02) for w0<1(approx)
# - three solutions (r_01, r_02, r_03) for 1(approx)<w0<1/4I0
# - two solutions (r_01=r_02, r_03) for w0=1/4I0
# - one solution (r_03) for w0>1/4I0

w0 = -1
I0 = 0.5

r0_num = fixed_point_solver(w0, I0, initial_guess = r_02(w0, I0))
print(f'Numerical Solution r0 = {r0_num}')

r_01_val = r_01(w0, I0) 
r_02_val = r_02(w0, I0) 
r_03_val = r_03(w0, I0) 
r_04_val = r_04(w0, I0) 
print(f'Theoretical Solution r0 = {r_02_val}')


r0_values = np.linspace(-5, 18, 1000)
phi_values = nonlinearity(w0 * r0_values + I0)

plt.figure(figsize=(10, 6))

cond_less_than_0 = w0 * r0_values + I0 < 0
cond_between_0_and_1 = (w0 * r0_values + I0 >= 0) & (w0 * r0_values + I0 <= 1)
cond_greater_than_1 = w0 * r0_values + I0 > 1

plt.fill_between(r0_values, -10, 15, where=cond_less_than_0, color='lightgray', alpha=0.3, label='$(w_0 \cdot r_0 + I_0) < 0$')
plt.fill_between(r0_values, -10, 15, where=cond_between_0_and_1, color='lightblue', alpha=0.3, label='$0 \leq (w_0 \cdot r_0 + I_0) \leq 1$')
plt.fill_between(r0_values, -10, 15, where=cond_greater_than_1, color='lightcoral', alpha=0.3, label='$(w_0 \cdot r_0 + I_0) > 1$')

plt.plot(r0_values, phi_values, label='$\phi(w_0 \cdot r_0 + I_0)$', color='blue')
plt.plot(r0_values, r0_values, label='$r_0 = r_0$', linestyle='--', color='red')

plt.scatter([r_01_val], [r_01_val], color='red', label=f'$r_{{01}}$: {r_01_val:.2f}', zorder=5) 
plt.scatter([r_02_val], [r_02_val], color='orange', label=f'$r_{{02}}$: {r_02_val:.2f}', zorder=5) 
plt.scatter([r_03_val], [r_03_val], color='purple', label=f'$r_{{03}}$: {r_03_val:.2f}', zorder=5) 
plt.scatter([r_04_val], [r_04_val], color='blue', label=f'$r_{{04}}$: {r_04_val:.2f}', zorder=5) 

plt.gca().set_facecolor('white')
plt.gcf().set_facecolor('white')
plt.xlabel('$r_0$', color='black')
plt.ylabel('Value', color='black')
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.legend()
plt.grid(color='gray')

plt.show()

