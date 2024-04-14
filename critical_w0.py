import numpy as np
from scipy.optimize import fsolve
from common_utils import derivative_nonlinearity, r_01, r_02, r_03
import matplotlib.pyplot as plt


I_0 = 1/4
equation1 = lambda w0: w0*derivative_nonlinearity(w0 * r_01(w0, I_0) + I_0) - 1 
equation2 = lambda w0: w0*derivative_nonlinearity(w0 * r_02(w0, I_0) + I_0) - 1 
equation3 = lambda w0: w0*derivative_nonlinearity(w0 * r_03(w0, I_0) + I_0) - 1 

w0 = np.linspace(-10,10, 100)

initial_guess = 0.5  
w0_solution1 = fsolve(equation1, initial_guess)
w0_solution2 = fsolve(equation2, initial_guess) # this is equal to 1/4*I0
w0_solution3 = fsolve(equation3, initial_guess) 

print(f'Max2: {w0[np.argmax(equation1(w0))]}')
print(f'Max3: {w0[np.argmax(equation3(w0))]}')

print(f"The solution for w0 is approximately: {w0_solution1[0]}")
print(f'Result at the solution: {equation2(w0_solution2)}')

plt.figure(figsize=(10, 6))  
plt.plot(w0, equation1(w0), 'b-')  
plt.plot(w0, equation2(w0), 'r-')  
plt.plot(w0, equation3(w0), 'g-')  
plt.plot(w0_solution1, equation1(w0_solution1), 'ro', markersize=3)  
plt.plot(w0_solution2, equation2(w0_solution2), 'bo', markersize=3)  
plt.plot(w0_solution3, equation3(w0_solution3), 'go', markersize=3)  
plt.show()  

