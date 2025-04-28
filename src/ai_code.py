import matplotlib.pyplot as plt
import numpy as np
from scipy import stats # For easily calculating SEM (stats.sem)


x_position, height_values, bar_color = 1, [10, 12, 11.5, 13, 10.8, 12.5, 9.8, 11.9], 'skyblue'

# --- Plotting Code (3 Lines) ---
mean_h = np.mean(height_values) # 1. Calculate mean
sem_h = np.std(height_values, ddof=1) / np.sqrt(len(height_values)) if len(height_values) > 1 else 0 # 2. Calculate SEM (Standard Error)
plt.bar(x_position, mean_h, yerr=sem_h, color=bar_color, capsize=5); plt.show() # 3. Plot bar with error and display