import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Your original data points (xp must be sorted)
xp = np.array([0, 5, 10, 15, 20])
fp = np.array([10, 50, 30, 80, 60])

# Points where you want to interpolate
x_new = np.linspace(-2, 22, 50)

# --- Method 1: Using scipy.interpolate.interp1d (Recommended) ---

# Create the 'previous' neighbor interpolation function
# bounds_error=False prevents errors for x_new outside xp's range
# fill_value="extrapolate" uses the first value for x < xp[0]
# and the last value for x > xp[-1], which is natural for 'previous'
f_previous = interp1d(xp, fp, kind='previous', bounds_error=False, fill_value="extrapolate")

# Alternatively, fill with NaN outside the bounds:
# f_previous_nan = interp1d(xp, fp, kind='previous', bounds_error=False, fill_value=np.nan)

# Apply the interpolation function
y_previous = f_previous(x_new)
# y_previous_nan = f_previous_nan(x_new)

# --- For Comparison: Original np.interp (Linear) ---
y_linear = np.interp(x_new, xp, fp)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(xp, fp, 'o', label='Original Data Points', markersize=8)
plt.step(xp, fp, where='post', label='True Step Function (for visualization)', linestyle='--', alpha=0.7) # Helper line shows ideal steps

plt.plot(x_new, y_previous, label="interp1d(kind='previous')")
plt.plot(x_new, y_linear, label="np.interp (linear)", linestyle=':', alpha=0.8)

# Example with NaN fill
# plt.plot(x_new, y_previous_nan, label="interp1d(kind='previous', fill_value=np.nan)", linestyle='-.')

plt.title('Interpolation Methods')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True)
plt.show()

# Print some specific values
print("x | interp1d('previous') | np.interp (linear)")
print("-" * 40)
for x_val in [2.5, 5.0, 7.5, 14.9, 15.0, 21.0, -1.0]:
    y_prev = f_previous(x_val)
    y_lin = np.interp(x_val, xp, fp)
    print(f"{x_val:<5} | {y_prev:<20} | {y_lin:<15}")