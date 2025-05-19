import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([2, 4, 5, 4, 6, 8, 7, 9, 10, 12])

# Create a figure and a single subplot
fig, ax = plt.subplots()

# Create the scatter plot on the subplot
ax.scatter(x_data, y_data, label='Data Points')

# Calculate the linear regression
slope, intercept = np.polyfit(x_data, y_data, 1)

# Generate points for the regression line
regression_x = np.linspace(min(x_data), max(x_data), 100)
regression_y = slope * regression_x + intercept

# Plot the linear regression line on the subplot
ax.plot(regression_x, regression_y, color='red', label=f'Linear Fit: y={slope:.2f}x + {intercept:.2f}')

# Customize and show the plot
ax.set_xlabel("Independent Variable")
ax.set_ylabel("Dependent Variable")
ax.set_title("Scatter Plot with Linear Regression Fit")
ax.legend()
ax.grid(True)

plt.show()