import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from common_utils import r_01, r_02, r_03, r_04, apply_mask, select_and_solve, derivative_nonlinearity

I0 = 0.75
N = 10000
bounds = (-5, 10)
w0 = np.linspace(bounds[0], bounds[1], N)
N = 10000
w0_values = np.linspace(-3, 5, N)

np.seterr(divide='ignore', invalid='ignore')

for func in [r_01, r_02, r_03, r_04]:  
     w0_filtered, r_filtered = apply_mask(w0_values, I0, func, tolerance=1e-8)
        
     stable = w0_filtered < 1 / derivative_nonlinearity(w0_filtered * r_filtered + I0)

     plt.plot(w0_filtered[stable], r_filtered[stable], 'black', linestyle='-', label='Stable' if 'Stable' not in plt.gca().get_legend_handles_labels()[1] else '')
     plt.plot(w0_filtered[~stable], r_filtered[~stable], 'red', label='Unstable' if 'Unstable' not in plt.gca().get_legend_handles_labels()[1] else '')

     w0_selected, r_num = select_and_solve(w0_filtered, r_filtered, I0, func)
     if len(w0_selected) > 0 and (func == r_02 or func == r_03):
          cond = len(w0_selected) if func == r_02 else len(w0_selected)
          plt.plot(w0_selected[:cond], r_num[:cond], 'o', alpha=0.75, markersize=5)


plt.xlabel('$w_0$')
plt.ylabel('$r_0$')
plt.title(f'Bifurcation Diagram at $I_0 = {I0:.2f}$')
plt.legend()
plt.grid(True)
plt.show()