import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from common_utils import dr_dt, r_01, r_02, r_03, nonlinearity, derivative_nonlinearity

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
        r0 = [func(w0_val, I_0)] 
        sol = solve_ivp(dr_dt, t_span, r0, args=(w0_val, I_0), t_eval=t_eval, method='RK45')
        r_num.append(sol.y[0, -1])
        
    return w0_selected, r_num

I_0 = 0
N = 10000
bounds = (-5, 10)
w0 = np.linspace(bounds[0], bounds[1], N)

equation1 = lambda w0: w0*derivative_nonlinearity(w0 * r_01(w0, I_0) + I_0) - 1 
equation2 = lambda w0: w0*derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0) - 1 
equation3 = lambda w0: w0*derivative_nonlinearity(w0 * r_03(w0, I_0) + I_0) - 1 

initial_guess = 0.5 
w0_solution1 = fsolve(equation1, initial_guess)[0] # there is something with w0[np.argmax(equation1(w0))], find it
w0_solution2 = fsolve(equation2, initial_guess)[0] # this is equal to 1/4*I0
w0_solution3 = fsolve(equation3, initial_guess)[0]

plt.figure(figsize=(10, 6))


for func, label, linestyle in [(r_01, '$r_{01}(w_0, I_0)$', '--'), 
                                          (r_02, '$r_{02}(w_0, I_0)$', '-'), 
                                          (r_03, '$r_{03}(w_0, I_0)$', '-.')]:
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

plt.axvline(x=w0_solution1, color='red', linestyle='--', label='Critical w0 for r01 = ' + f'{w0_solution1}')
plt.axvline(x=w0_solution2, color='red', linestyle='--', label='Critical w0 for r02 = ' + f'{w0_solution2}')
#plt.axvline(x=w0_solution3, color='red', linestyle='--', label='Critical w0 for r03 = ' + f'{w0_solution3}')
plt.xlabel('$w_0$')
plt.ylabel('Response')
plt.title('Filtered Response Functions with Numerical Solutions')
plt.legend()
plt.grid(True)
plt.show()
