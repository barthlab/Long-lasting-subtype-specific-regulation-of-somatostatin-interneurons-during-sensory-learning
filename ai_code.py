import numpy as np
import matplotlib.pyplot as plt # Optional: for visualization

# --- Example Data ---
# Let's create some sample data resembling a peak
# (Replace this with your actual x and y data)
x = np.linspace(0, 10, 101) # 101 points from 0 to 10
# A skewed peak (e.g., a skewed Gaussian or log-normal shape)
# Using a simple Gaussian for demonstration:
mean1 = 4.5
mean2 = 2
sigma1 = 1.5
sigma2 = .5
y = 0.5*np.exp(-((x - mean1)**2) / (2 * sigma1**2)) + np.exp(-((x - mean2)**2) / (2 * sigma2**2))
# Optional: add some noise
# y += np.random.normal(0, 0.05, size=x.shape)
# y = np.maximum(y, 0) # Ensure y is non-negative if needed for COM interpretation

# --- Calculation ---

# 1. Calculate the first moment (integral of x*y dx)
# Ensure y is non-negative if interpreting as 'mass' or 'intensity'
# If y can be negative, the physical interpretation might change,
# but the calculation is the same. Consider y_processed = np.maximum(y, 0)
# if negative values don't make sense for weighting.
first_moment = np.trapz(y * x, x)

# 2. Calculate the total area (AUC or total 'mass')
total_area = np.trapz(y, x)

# 3. Calculate the Center of Mass
# Add a check to prevent division by zero if the area is zero
if total_area == 0:
    center_of_mass = np.nan # Or handle as appropriate (e.g., mean of x)
    print("Warning: Total area is zero. Cannot calculate center of mass.")
else:
    center_of_mass = first_moment / total_area

# --- Output ---
print(f"X data range: {x.min()} to {x.max()}")
print(f"Peak Y value: {y.max():.4f}")
print(f"Peak X position: {x[np.argmax(y)]:.4f}")
print(f"Area Under Curve (AUC): {total_area:.4f}")
print(f"First Moment (Integral of x*y dx): {first_moment:.4f}")
print(f"Center of Mass (Weighted X): {center_of_mass:.4f}")


# --- Optional: Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Data (y vs x)', color='blue')
plt.fill_between(x, y, color='lightblue', alpha=0.4, label=f'AUC = {total_area:.2f}')

peak_x_pos = x[np.argmax(y)]
plt.axvline(peak_x_pos, color='red', linestyle='--', label=f'Peak Position = {peak_x_pos:.2f}')

if not np.isnan(center_of_mass):
 plt.axvline(center_of_mass, color='green', linestyle='-', linewidth=2, label=f'Center of Mass = {center_of_mass:.2f}')

plt.title('Data, Peak Position, and Center of Mass')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()