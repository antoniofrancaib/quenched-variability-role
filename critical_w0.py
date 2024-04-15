import numpy as np
from scipy.optimize import fsolve
from common_utils import derivative_nonlinearity, r_01, r_02, r_03
import matplotlib.pyplot as plt


I_0 = 1/4

equation = lambda w0: 1 / derivative_nonlinearity(w0 * r_01(w0, I_0) + I_0) 

w0 = np.linspace(-10,10, 100)

plt.figure(figsize=(10, 6))  
plt.plot(w0, equation(w0), 'b-')  
plt.plot(w0, w0, 'r-')  
plt.show()  

