import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from common_utils import nonlinearity

def phi(x, r0=10, beta=5.0):
    return r0 / (1 + np.exp(-beta * x))

def dr_dt(t, r, w0, I_0):
    return -r + nonlinearity(w0 * r + I_0)

def d_phi(x, r0, beta):
    exp_term = np.exp(-beta * x)
    derivative = (beta * r0 * exp_term) / ((1 + exp_term) ** 2)
    return derivative

w0 = -1
I_0 = 1/4

r_01 = lambda w0, I_0: (1 - 2 * w0 * I_0 + np.sqrt(1 - 4 * w0 * I_0)) / (2 * w0**2)
r_04 = lambda w0, I_0: 2*w0 - 2*np.sqrt(w0**2 + I_0 - (3/4))

A = +np.sqrt(w0**2+I_0-(3/4))
print(f'A={A}')
r0 = [0]#[2*w0+2*A] #[(1-2*w0*I0+A)/(2*np.power(w0,2))] 
check = r0[0] > (1-I_0)/w0
print(check)
print(f'r0={r0[0]}')

t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)  

sol = solve_ivp(dr_dt, t_span, r0, args=(w0, I_0), t_eval=t_eval, method='RK45')

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label=f'w0={w0}, I={I_0}')
plt.title('Solution of the differential equation')
plt.xlabel('Time t')
plt.ylabel('r(t)')
plt.legend()
plt.grid(True)
plt.show()

print(f'Solution: {sol.y[0][-1]}')