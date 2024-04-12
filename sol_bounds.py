import numpy as np 
import matplotlib.pyplot as plt
from common_utils import nonlinearity, derivative_nonlinearity, r_01, r_03
from scipy.optimize import fsolve

I0=3/4
x = np.linspace(-10, 5, 1000)

"""
plt.figure(figsize=(10, 6))  
plt.plot(x, r_01(x, I0), 'r-')  
plt.plot(x, r_03(x, I0), 'b-')  
plt.axvline(x=1/(4*I0), color='red', linestyle='--')
plt.axvline(x=np.sqrt((3/4)-I0), color='blue', linestyle='--')
plt.show()  
"""


# SOL 1 BOUNDS ARE 
w0_values = np.linspace(-10, 5, 1000)
equation1 = lambda w0: (1+I0*(1-2*w0)+np.sqrt(1-4*w0*I0))/(2*w0) 
lower_bound1 = fsolve(lambda w0: (1+I0*(1-2*w0)+np.sqrt(1-4*w0*I0))/(2*w0) - 1, 1/(4*I0) if not None else 1)[0] 
print(lower_bound1)

plt.figure(figsize=(10, 6))  
plt.plot(x, equation1(w0_values), 'r-')  
plt.plot(lower_bound1, equation1(lower_bound1), 'ro')  
plt.axvline(x=1/(4*I0), color='red', linestyle='--')
plt.axhline(y=1, color='red', linestyle='--')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()  

# SOL 2 BOUNDS ARE [lower_bound2, 1/4*I0]
w0_values = np.linspace(-10, 5, 10000)
equation2 = lambda w0: (1+I0*(1-2*w0)-np.sqrt(1-4*w0*I0))/(2*w0) 
lower_bound2 = fsolve(lambda w0: (1+I0*(1-2*w0)-np.sqrt(1-4*w0*I0))/(2*w0) - 1, 0.1)[0] 
print(f'LOWER BOUND FOR SOL 2: {lower_bound2}')

plt.figure(figsize=(10, 6))  
plt.plot(w0_values, equation2(w0_values), 'r-')  
plt.plot(lower_bound2, equation2(lower_bound2), 'ro')  
#plt.axvline(x=1/(4*I0), color='red', linestyle='--')
plt.axhline(y=1, color='red', linestyle='--')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()  
