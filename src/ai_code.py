import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
num_point = 50 # Example width

# --- Create Sample Data ---
# Replace this with your actual boolean numpy array
bool_array = np.random.choice([True, False], size=num_point, p=[0.6, 0.4])
# Example: bool_array = np.array([True, False, True, True, False, ...])

# --- Reshape for imshow ---
# imshow expects 2D data. Reshape (num_point,) -> (1, num_point)
image_data = bool_array[np.newaxis, :]

# --- Define Coordinates for the Rectangle ---
left_coord = -num_point / 2
right_coord = num_point / 2
bottom_coord = 0.0
top_coord = 1.0
plot_extent = (left_coord, right_coord, bottom_coord, top_coord)

# --- Plotting ---
# Create a figure and axes for the overall plot
fig, ax = plt.subplots(figsize=(10, 4)) # Adjust figsize for the overall plot

# Display the boolean array data as an image at the specified coordinates
# extent=(left, right, bottom, top) defines data coordinates
# origin='lower': Places the [0,0] index of the array at the bottom-left corner defined by extent
# aspect='auto': Stretches the image to fill the extent rectangle in data coordinates.
#                Individual squares might not be visually square depending on axis scaling.
im = ax.imshow(
    image_data,
    cmap='Greys',
    interpolation='nearest',
    extent=plot_extent,
    origin='lower', # Important when using extent for y-coordinates 0 to 1
    aspect='auto'   # Stretch to fill the defined extent
)

# --- Customize Appearance ---
ax.set_title(f'Boolean Array Plotted from x={left_coord} to {right_coord}, y={bottom_coord} to {top_coord}')
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# --- Set Axis Limits ---
# **Crucial**: Set the limits wider than the rectangle to see its position
# Adjust these limits based on other elements you might have on the plot
ax.set_xlim(left_coord - 10, right_coord + 10) # Example padding
ax.set_ylim(bottom_coord - 1, top_coord + 1)   # Example padding

# Optional: Add a grid to see coordinates clearly
ax.grid(True, linestyle='--', alpha=0.6)

# Optional: Force equal aspect ratio for the axes
# If you uncomment this, one unit on the x-axis will look the same
# length as one unit on the y-axis. This might make the individual
# boolean squares look more square, but could add whitespace or change
# the overall plot shape significantly.
# ax.set_aspect('equal', adjustable='box')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()