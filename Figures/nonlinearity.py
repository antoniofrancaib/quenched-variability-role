import numpy as np
import matplotlib.pyplot as plt

def nonlinearity(x):
    phi_x = np.zeros_like(x)
    # Apply the square power for values between 0 and 1
    phi_x[(x >= 0) & (x <= 1)] = x[(x >= 0) & (x <= 1)] ** 2
    # Apply the modified square root for values greater than 1
    phi_x[x > 1] = 2 * np.sqrt(x[x > 1] - (3 / 4)) 
    return phi_x

# Generate x values from -1 to 3 for plotting
x = np.linspace(-1, 3, 400)
y = nonlinearity(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Nonlinearity $\phi(x)$', color='cyan')
plt.title('Plot of the Nonlinearity Function', color='white')
plt.xlabel('x', color='white')
plt.ylabel('$\phi(x)$', color='white')
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.grid(color='grey', linestyle='--', linewidth=0.5)
plt.legend()

# Invert colors for futuristic theme
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('black')
plt.tick_params(colors='white', which='both')  # changing tick color to white

# Show the plot
plt.show()
