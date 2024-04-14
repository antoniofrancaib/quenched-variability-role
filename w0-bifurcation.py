import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common_utils import dr_dt, r_01, r_02, r_03, r_04, find_critical_w0, apply_mask, select_and_solve

I_0 = 0.15
N = 10000
bounds = (-5, 10)
w0 = np.linspace(bounds[0], bounds[1], N)

critical_r01 = find_critical_w0(r_01, I_0)
critical_r02 = find_critical_w0(r_02, I_0)
critical_r03 = find_critical_w0(r_03, I_0)
critical_r04 = find_critical_w0(r_04, I_0)

plt.figure(figsize=(10, 6))

for func, label, linestyle in [(r_01, '$r_{01}(w_0, I_0)$', '--'), 
                               (r_02, '$r_{02}(w_0, I_0)$', '-'), 
                               (r_03, '$r_{03}(w_0, I_0)$', '-.'),
                               (r_04, '$r_{04}(w_0, I_0)$', ':')]:  
    w0_filtered, r_filtered = apply_mask(w0, I_0, func)
    plt.plot(w0_filtered, r_filtered, label=label, linestyle=linestyle)
    
    w0_selected, r_num = select_and_solve(w0_filtered, r_filtered, I_0, func)
    if len(w0_selected) > 0:
         plt.plot(w0_selected, r_num, '.', label=f'Numerical {label}', alpha=0.75, markersize=5)

if (I_0 != 0) & (I_0 < 10):
    plt.axvline(x=1/(4*I_0), color='blue', linestyle='--', label='$\\frac{1}{4I_0} = ' + f'{1/(4*I_0):.2f}$' + ' Vertical Line')

if I_0 < 3/4:
    positive_sqrt_term = np.sqrt((3/4) - I_0)
    negative_sqrt_term = -np.sqrt((3/4) - I_0)
        
    plt.axvline(x=positive_sqrt_term, color='green', linestyle='--', label=f'$+\\sqrt{{3/4 - I_0}} = {positive_sqrt_term:.2f}$ Vertical Line')
    plt.axvline(x=negative_sqrt_term, color='blue', linestyle='--', label=f'$-\\sqrt{{3/4 - I_0}} = {negative_sqrt_term:.2f}$ Vertical Line')

plt.axvline(x=critical_r01, color='purple', linestyle='--', label='Critical w0 for r01 = ' + f'{critical_r01}')
plt.axvline(x=critical_r02, color='red', linestyle='--', label='Critical w0 for r02 = ' + f'{critical_r02}')
plt.axvline(x=critical_r03, color='green', linestyle='--', label='Critical w0 for r03 = ' + f'{critical_r03}')
plt.axvline(x=critical_r04, color='brown', linestyle='--', label='Critical w0 for r03 = ' + f'{critical_r04}')

plt.xlabel('$w_0$')
plt.ylabel('Response')
plt.title('Filtered Response Functions with Numerical Solutions')
plt.legend()
plt.grid(True)
plt.show()
