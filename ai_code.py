import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3.5, 2.8, 4.2, 3.1])
y_error = np.array([0.3, 0.5, 0.4, 0.6, 0.35])

plt.figure(figsize=(8, 6))

plt.errorbar(x, y, yerr=y_error, fmt=' ', capsize=5, elinewidth=1.5, color='royalblue')

plt.title('Error Bar Plot without Bars')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
