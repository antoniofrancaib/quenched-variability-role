import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from common_utils import r_01, r_02, r_03

N = 1000
w0_values = np.linspace(-5, 5, N)
I0_values = np.linspace(0, 2, N)

W0, I0 = np.meshgrid(w0_values, I0_values)

# For r_01
R_01 = r_01(W0, I0)
mask_01 = (-I0/W0 <= R_01) & (R_01 <= (1-I0)/W0)
R_01_filtered = np.where(mask_01, R_01, np.nan)

# For r_02
R_02 = r_02(W0, I0)
mask_02 = (-I0/W0 <= R_02) & (R_02 <= (1-I0)/W0)
R_02_filtered = np.where(mask_02, R_02, np.nan)

# For r_03
R_03 = r_03(W0, I0)
mask_03 = (R_03 > (1-I0)/W0)
R_03_filtered = np.where(mask_03, R_03, np.nan)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot for r_01
surf1 = ax.plot_surface(W0, I0, R_01_filtered, cmap='viridis', edgecolor='none', alpha=0.5)

# Plot for r_02
surf2 = ax.plot_surface(W0, I0, R_02_filtered, cmap='inferno', edgecolor='none', alpha=0.5)

# Plot for r_03
surf3 = ax.plot_surface(W0, I0, R_03_filtered, cmap='plasma', edgecolor='none', alpha=0.5)

ax.set_xlabel('$w_0$')
ax.set_ylabel('$I_0$')
ax.set_zlabel('Values')
plt.title('Filtered 3D Plots of $r_{01}(w_0, I_0)$, $r_{02}(w_0, I_0)$, and $r_{03}(w_0, I_0)$')
plt.colorbar(surf1, ax=ax, shrink=0.5, aspect=5, label='$r_{01}(w_0, I_0)$')
plt.colorbar(surf2, ax=ax, shrink=0.5, aspect=5, label='$r_{02}(w_0, I_0)$')
plt.colorbar(surf3, ax=ax, shrink=0.5, aspect=5, label='$r_{03}(w_0, I_0)$')

plt.show()
