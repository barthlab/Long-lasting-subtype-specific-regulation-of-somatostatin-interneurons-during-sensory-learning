import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm # Import colormaps
from scipy.interpolate import splprep, splev

# --- Configuration ---
NUM_TRACES = 10      # How many example traces to generate (increased for better viz)
MIN_LEN = 10         # Minimum length of a trace
MAX_LEN = 20         # Maximum length of a trace
SMOOTHING_FACTOR = 1 # Controls smoothness (0=interpolate, >0=smooth).
ARROW_HEAD_WIDTH = 0.15 # Adjust arrow appearance
ARROW_HEAD_LENGTH = 0.2 # Adjust arrow appearance
NUM_SMOOTH_POINTS = 100 # How many points to generate for the smooth curve plot
LINE_WIDTH = 2.5      # Width of the plotted lines

# Define Colormaps for the two groups
CMAP_GROUP1 = 'Blues'   # Example: Light blue to dark blue
CMAP_GROUP2 = 'Oranges' # Example: Light orange to dark orange
# Other options: 'Greens', 'Reds', 'Purples', 'Greys'
# Or sequential: 'viridis'/'plasma' vs 'magma'/'inferno'

# --- 1. Generate Sample Data (Replace with your actual list of arrays) ---
# (Using the same generation code as before for consistency)
np.random.seed(42) # for reproducibility
trace_list = []
for i in range(NUM_TRACES):
    length = np.random.randint(MIN_LEN, MAX_LEN + 1)
    t = np.linspace(0, 2 * np.pi, length)
    base_x = np.cos(t) * (i + 1) * 0.3 + np.random.randn() * 2 # Adjusted scale slightly
    base_y = np.sin(t) * (i + 1) * 0.3 + np.random.randn() * 2 # Adjusted scale slightly
    noise_x = np.random.randn(length) * 0.3
    noise_y = np.random.randn(length) * 0.3
    wiggles = np.sin(t * np.random.uniform(2,5)) * 0.4 # Varied wiggles
    x = base_x + noise_x + wiggles * np.random.choice([-1, 1])
    y = base_y + noise_y - wiggles * np.random.choice([-1, 1])
    trace = np.vstack((x, y))
    trace_list.append(trace)

# --- 2. Setup Figure with Axes for Plot and Colorbars ---
# Create 1 row, 3 columns: [main plot axis | colorbar 1 axis | colorbar 2 axis]
# Adjust width_ratios: plot takes most space, colorbars are narrow.
fig, axes = plt.subplots(1, 3, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [15, 1, 1]})

main_ax = axes[0] # Axis for the trace plots
cbar1_ax = axes[1] # Axis for the first colorbar
cbar2_ax = axes[2] # Axis for the second colorbar

print(f"Using Smoothing Factor (s): {SMOOTHING_FACTOR}")
print(f"Group 1 Colormap: {CMAP_GROUP1}")
print(f"Group 2 Colormap: {CMAP_GROUP2}")
print("-" * 30)

# Get the chosen colormap objects
cmap1 = cm.get_cmap(CMAP_GROUP1)
cmap2 = cm.get_cmap(CMAP_GROUP2)

# Define the normalization (maps path parameter 0-1 to color range 0-1)
# This can be shared as it represents progress along *each individual* curve
norm = Normalize(vmin=0, vmax=1)

# --- 3. Process and Plot Each Trace with Grouped Gradient Colors ---
for i, trace_arr in enumerate(trace_list):
    x_orig = trace_arr[0, :]
    y_orig = trace_arr[1, :]
    n_points = len(x_orig)

    print(f"Processing Trace {i+1} with {n_points} points.")

    if n_points < 4:
        print(f"  Skipping Trace {i+1}: Too few points ({n_points}).")
        continue

    # Determine group and select colormap
    # Simple grouping: even index = group 1, odd index = group 2
    if i % 2 == 0:
        current_cmap = cmap1
        group_name = "Group 1"
    else:
        current_cmap = cmap2
        group_name = "Group 2"
    # print(f"  Assigned to {group_name}") # Optional debug print

    try:
        # --- Smoothing ---
        tck, u = splprep([x_orig, y_orig], s=SMOOTHING_FACTOR, k=3, per=0)
        u_new = np.linspace(0, 1, NUM_SMOOTH_POINTS) # Normalized parameter [0, 1]
        x_smooth, y_smooth = splev(u_new, tck)

        # --- Plotting with Gradient Line ---
        points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection with the group's colormap and shared normalization
        lc = LineCollection(segments, cmap=current_cmap, norm=norm)
        lc.set_array(u_new[:-1]) # Color based on parameter at segment start
        lc.set_linewidth(LINE_WIDTH)
        main_ax.add_collection(lc) # Add to the main plotting axis

        # --- Arrow (colored based on the end of the gradient for its group) ---
        if len(x_smooth) >= 2:
            x_start = x_smooth[-2]
            y_start = y_smooth[-2]
            dx = x_smooth[-1] - x_start
            dy = y_smooth[-1] - y_start

            # Get the arrow color from the correct colormap at the end parameter (1.0)
            arrow_color = current_cmap(norm(u_new[-1])) # or current_cmap(1.0)

            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                 main_ax.arrow(x_start, y_start, dx, dy,
                               head_width=ARROW_HEAD_WIDTH,
                               head_length=ARROW_HEAD_LENGTH,
                               fc=arrow_color, ec=arrow_color,
                               length_includes_head=True,
                               zorder=5) # Draw on top
            else:
                 print(f"  Warning: Arrow for Trace {i+1} has near-zero length. Skipping arrow.")

    except Exception as e:
        print(f"  Error processing Trace {i+1} ({group_name}): {e}")

# --- 4. Final Plot Adjustments & Colorbars ---
main_ax.set_xlabel("X Coordinate")
main_ax.set_ylabel("Y Coordinate")
main_ax.set_title(f"Smoothed Traces by Group ({CMAP_GROUP1} / {CMAP_GROUP2})")
main_ax.grid(True, linestyle='--', alpha=0.6)
main_ax.set_aspect('equal', adjustable='box')
# Auto-ranging based on plotted data
main_ax.autoscale_view()

# --- Create Colorbars ---
# Create dummy ScalarMappable objects to link the colormaps to the colorbar instances
sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
sm1.set_array([]) # Necessary dummy data
sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
sm2.set_array([])

# Add colorbars to their dedicated axes
cbar1 = fig.colorbar(sm1, cax=cbar1_ax)
cbar1.set_label('Group 1 Path Param (0=start, 1=end)')

cbar2 = fig.colorbar(sm2, cax=cbar2_ax)
cbar2.set_label('Group 2 Path Param (0=start, 1=end)')

# Adjust layout to prevent labels/titles overlapping
plt.tight_layout(rect=[0, 0, 0.95, 1]) # Leave space on right, adjust as needed

plt.show()