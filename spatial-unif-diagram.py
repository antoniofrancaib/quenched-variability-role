import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from common_utils import nonlinearity

def dr_dt(t, r, w0, I_0):
    return -r + nonlinearity(w0 * r + I_0)

# IMPONER CONDITIONAL INEQUALITIES CON CODIGO PARA HACERLO TODO MAS PRECISO 

I_0 = 1/2 # try 1/16

r_01 = lambda w0, I_0: (1 - 2 * w0 * I_0 + np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2) # UNSTABLE
r_02 = lambda w0, I_0: (1 - 2 * w0 * I_0 - np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2) # STABLE 
r_03 = lambda w0, I_0: 2*w0 + 2*np.sqrt(w0**2 + I_0 - (3/4)) # STABLE 
r_04 = lambda w0, I_0: 2*w0 - 2*np.sqrt(w0**2 + I_0 - (3/4))

w01 = np.linspace(1, 5, 100) if I_0 == 0 else np.linspace(1, 1/(4*I_0), 100) if I_0 < 1/4 else None # 0.88 funciona como lower bound
w02 = np.linspace(-3, 5, 100) if I_0 == 0 else np.linspace(-3, 1/(4*I_0), 100)
w03 = np.linspace(0, 5, 100) #if I_0 > 1 else np.linspace(-5, 0, 100) if I_0 < 1 else None

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



plt.title('Plot of r_0 as a function of w0 with Vertical Line at $1/(4I_0)$')
plt.xlabel('w0')
plt.ylabel('r_0')

plt.legend(loc='best')

plt.grid(True)
plt.show()
