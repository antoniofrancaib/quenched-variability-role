import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from common_utils import nonlinearity, derivative_nonlinearity, r_01, r_02, r_03, r_04

def dr_dt(t, r, w0, I_0):
    return -r + nonlinearity(w0 * r + I_0)

# IMPONER CONDITIONAL INEQUALITIES CON CODIGO PARA HACERLO TODO MAS PRECISO 
# how can i plot critical w0 if it depends on 1/derivative(w0)

N = 10000
I_0 = 0.4 # try 1/2 - 1/16

equation1 = lambda w0: w0*derivative_nonlinearity(w0 * r_01(w0, I_0) + I_0) - 1 
equation2 = lambda w0: w0*derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0) - 1 
equation3 = lambda w0: w0*derivative_nonlinearity(w0 * r_03(w0, I_0) + I_0) - 1 

w0 = np.linspace(-10,10, N)

initial_guess = 0.5 
w0_solution1 = fsolve(equation1, initial_guess)[0] # there is something with w0[np.argmax(equation1(w0))], find it
w0_solution2 = fsolve(equation2, initial_guess)[0] # this is equal to 1/4*I0
w0_solution3 = fsolve(equation3, initial_guess)[0]

w01 = np.linspace(w0_solution1, 5, N) if I_0 == 0 else np.linspace(w0_solution1, 1/(4*I_0), N) if I_0 < 1/4 else None # 0.88 funciona como lower bound
w02 = np.linspace(-10, 5, N) if I_0 == 0 else np.linspace(-3, 1/(4*I_0), N)
w03 = np.linspace(w0_solution3, 5, N) #if I_0 > 1 else np.linspace(-5, 0, 100) if I_0 < 1 else None

r_01_num = []
r_02_num = []
r_03_num = []
r_04_num = []

t_span = (0, 50)
t_eval = np.linspace(*t_span, 1000)  

if w01 is not None: 
    w01_selected = np.linspace(w01[0], w01[-1], 5) 
    for w0 in w01_selected:
        r0 = [r_01(w0, I_0)]
        sol01 = solve_ivp(dr_dt, t_span, r0, args=(w0, I_0), t_eval=t_eval, method='RK45')  

        r_01_num.append(sol01.y[0, -1]) # get the last value (convergence)

if w02 is not None:
    w02_selected = np.linspace(w02[0], w02[-1], 5)
    for w0 in w02_selected:
        r0 = [r_02(w0, I_0)]
        if not np.isnan(r0[0]) and np.isfinite(r0[0]):
            sol02 = solve_ivp(dr_dt, t_span, r0, args=(w0, I_0), t_eval=t_eval, method='RK45', rtol=1e-5)
            r_02_num.append(sol02.y[0, -1])  # get the last value (convergence)
        else:
            r_02_num.append(np.nan)


if w03 is not None:
    w03_selected = np.linspace(w03[0], w03[-1], 5)
    print(w03_selected)
    for w0 in w03_selected:
        r0 = [r_03(w0, I_0)] 

        if not np.isnan(r0[0]) and np.isfinite(r0[0]):
            sol03 = solve_ivp(dr_dt, t_span, r0, args=(w0, I_0), t_eval=t_eval, method='RK45', rtol=1e-5)
            r_03_num.append(sol03.y[0, -1])  # get the last value (convergence)
        else:
            r_03_num.append(np.nan)
  
    w04_selected = np.linspace(w03[0], w03[-1], 5)  # Using w03 again as per your code snippet
    for w0 in w04_selected:
        r0 = [r_04(w0, I_0)] 
        print(f'4: {r0}')

        if not np.isnan(r0[0]) and np.isfinite(r0[0]):
            sol04 = solve_ivp(dr_dt, t_span, [r_04(w0, I_0)], args=(w0, I_0), t_eval=t_eval, method='RK45', rtol=1e-5)
            r_04_num.append(sol04.y[0, -1])  # get the last value (convergence)

        else:
            r_04_num.append(np.nan)


plt.figure(figsize=(10, 6))

if w01 is not None:
    plt.plot(w01, r_01(w01, I_0), label='$r_{01}(w_0, I_0) = \\frac{1 - 2w_0I_0 + \\sqrt{1 - 4w_0I_0}}{2w_0^2}$', linestyle='--')
    plt.plot(w01_selected, r_01_num, 'r.', alpha=0.75, markersize=10, label='Numerical $r_{01}$')

if w02 is not None:
    plt.plot(w02, r_02(w02, I_0), label='$r_{02}(w_0, I_0) = \\frac{1 - 2w_0I_0 - \\sqrt{1 - 4w_0I_0}}{2w_0^2}$')
    plt.plot(w02_selected, r_02_num, 'g.', alpha=0.75, markersize=10, label='Numerical $r_{02}$')

if w03 is not None:
    plt.plot(w03, r_03(w03, I_0), label='$r_{03}(w_0, I_0) = 2w_0 + 2\\sqrt{w_0^2 + I_0 - \\frac{3}{4}}$')
    plt.plot(w03_selected, r_03_num, 'b.', alpha=0.75, markersize=10, label='Numerical $r_{03}$')
    
    """plt.plot(w03, r_04(w03, I_0), label='$r_{04}(w_0, I_0) = 2w_0 - 2\\sqrt{w_0^2 + I_0 - \\frac{3}{4}}$')
    plt.plot(w04_selected, r_04_num, 'y.', alpha=0.75, markersize=10, label='Numerical $r_{04}$')"""

if I_0 != 0:
    plt.axvline(x=1/(4*I_0), color='red', linestyle='--', label='$\\frac{1}{4I_0} = ' + f'{1/(4*I_0):.2f}$' + ' Vertical Line')


"""plt.axvline(x=w0_solution1, color='red', linestyle='--', label='Critical w0 for r01 = ' + f'{w0_solution1}')
plt.axvline(x=w0_solution2, color='red', linestyle='--', label='Critical w0 for r02 = ' + f'{w0_solution2}')
plt.axvline(x=w0_solution3, color='red', linestyle='--', label='Critical w0 for r03 = ' + f'{w0_solution3}')"""

plt.title('Bifurcation Diagram: r_0 as a function of w0')
plt.xlabel('w0')
plt.ylabel('r_0')

plt.legend(loc='best')

plt.grid(True)
plt.show()
