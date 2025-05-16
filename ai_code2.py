import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm
from scipy.stats import gaussian_kde # For manual KDE calculation

# --- Configuration ---
num_rows = 5
num_cols = 10
row_labels = [f"Series {chr(65+i)}" for i in range(num_rows)] # Example: Series A, B, ...

# Get a colormap (e.g., 'tab10' for distinct colors)
cmap = matplotlib.colormaps.get_cmap('tab10')
row_colors = [cmap(i) for i in range(num_rows)] # Ensure enough distinct colors if num_rows > 10

# --- Create Figure and Subplots ---
fig, axs = plt.subplots(num_rows, num_cols,
                        figsize=(num_cols * 2.5, num_rows * 1.3), # Adjust for visual clarity
                        sharex=True) # Share X axis among all subplots

# --- Global Y-axis Title ---
fig.text(0.02, 0.5, 'Category Group', va='center', ha='center', rotation='vertical', fontsize=12)

# --- Loop Through Subplots to Create Violins ---
for r in range(num_rows):
    for c in range(num_cols):
        ax = axs[r, c]
        # Y-coordinate for the violin's baseline within the subplot's [0,1] y-range
        violin_center_y = 0.5

        # 1. Generate/Load Sample Data for this specific violin
        # Replace this with your actual data for the (r, c) violin plot
        data_loc = np.random.uniform(15, 85)
        data_scale = np.random.uniform(5, 20)
        n_points = np.random.randint(80, 250)
        current_data = np.random.normal(loc=data_loc, scale=data_scale, size=n_points)
        current_data = np.clip(current_data, 0, 100) # Ensure data is within [0, 100]
        current_data = current_data[~np.isnan(current_data)] # Remove any NaNs

        # Handle cases with insufficient data for KDE
        if len(np.unique(current_data)) < 2 or len(current_data) < 2:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=9, color='grey')
            # Apply basic styling for consistency even if N/A
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            # Still draw the base line for the "slot"
            ax.axhline(violin_center_y, color='black', lw=1.0, zorder=1)
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.yaxis.grid(True, linestyle='-', color='#dddddd', linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if r == num_rows - 1:
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.tick_params(axis='x', labelsize=9, rotation=0, length=3)
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('black')
                ax.spines['bottom'].set_linewidth(1.0)
            else:
                ax.set_xticks([])
            if c == 0:
                 ax.text(-0.08, violin_center_y, row_labels[r],
                         transform=ax.transAxes, ha='right', va='center',
                         fontsize=10, clip_on=False)
            continue # Skip to the next subplot

        # 2. Calculate Kernel Density Estimate (KDE)
        try:
            kde = gaussian_kde(current_data)
            # Optional: Adjust bandwidth factor if desired (e.g., kde.factor * 0.75)
            # kde.set_bandwidth(bw_method=kde.factor / 1.5) # Example adjustment
        except Exception: # Catch potential errors in KDE calculation with tricky data
            # Fallback for KDE error (similar to N/A case)
            ax.text(0.5, 0.5, "KDE Err", ha="center", va="center", transform=ax.transAxes, fontsize=8, color='red')
            # (Apply basic styling as in the N/A case above)
            ax.set_xlim(0, 100); ax.set_ylim(0, 1)
            ax.axhline(violin_center_y, color='black', lw=1.0, zorder=1)
            # ... (rest of N/A styling)
            continue

        # Create a range of x values for plotting the KDE curve
        # Ensure x_plot_kde covers the data range; extend slightly for better visuals if needed
        data_min, data_max = np.min(current_data), np.max(current_data)
        if data_min == data_max: # Handle case where all data points are the same
             x_plot_kde = np.array([data_min - 0.1, data_min, data_min + 0.1]) # Create small range
             x_plot_kde = np.clip(x_plot_kde, 0, 100)
        else:
             x_plot_kde = np.linspace(data_min, data_max, 100) # 100 points for a smooth curve

        density_kde = kde(x_plot_kde)

        # Scale density to fit as the "height" of the violin half.
        # max_half_width determines how "fat" the violin is in the subplot's y-units.
        # E.g., if subplot ylim is [0,1] and violin_center_y is 0.5, max_half_width=0.4
        # means the violin's curve will go up to 0.5 + 0.4 = 0.9.
        max_half_width = 0.4
        if np.max(density_kde) > 1e-9: # Avoid division by zero or tiny numbers
            scaled_density = density_kde * (max_half_width / np.max(density_kde))
        else:
            scaled_density = np.zeros_like(density_kde) # Flat if no density

        # 3. Plot the Half-Violin Components
        # a. Plot the FILLED AREA of the half-violin with NO EDGE
        # Path for the filled area: follows the KDE curve on top, then flat along the center line.
        fill_x_coords = np.concatenate([x_plot_kde, x_plot_kde[::-1]])
        fill_y_coords = np.concatenate([violin_center_y + scaled_density,
                                        np.full_like(x_plot_kde, violin_center_y)[::-1]]) # Flat base

        ax.fill(fill_x_coords, fill_y_coords,
                facecolor=row_colors[r],
                alpha=0.75,          # Adjust transparency as needed
                edgecolor='none',    # CRUCIAL: No edge for the filled polygon
                zorder=2)            # zorder places it above grid, below lines

        # b. Plot the black OUTLINE for the CURVED part only
        ax.plot(x_plot_kde, violin_center_y + scaled_density,
                color='black',
                linewidth=1.0,       # Adjust line thickness
                zorder=3)            # Ensure line is on top of fill

        # c. Plot the black horizontal LINE for the BASE of the violin
        ax.axhline(violin_center_y, color='black', lw=1.0, zorder=3)


        # 4. Customize Subplot Appearance (axes, ticks, labels, grid)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

        # Light grey horizontal grid lines in the background
        ax.set_yticks([0.25, 0.5, 0.75]) # Example positions for grid lines
        ax.yaxis.grid(True, linestyle='-', color='#dddddd', linewidth=0.7, zorder=0) # zorder=0 for background
        ax.set_axisbelow(True) # Ensures grid is drawn behind plot elements

        # Remove y-tick labels as they are internal to subplot
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0) # Hide y-tick marks

        # Spines: remove all by default, then add back selectively
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # X-axis ticks and labels only for the bottom row
        if r == num_rows - 1:
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(axis='x', labelsize=9, rotation=0, length=3)
            # Draw the bottom spine for the plots in the last row
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1.0)
        else:
            ax.set_xticks([]) # Remove x-ticks for non-bottom rows
            ax.set_xticklabels([])

        # Add Row Labels to the left of the first column
        if c == 0:
            ax.text(-0.08, violin_center_y, row_labels[r],
                    transform=ax.transAxes,
                    ha='right', va='center', fontsize=10, clip_on=False)

# --- Adjust Layout ---
fig.subplots_adjust(left=0.12, bottom=0.08, right=0.98, top=0.95, wspace=0.15, hspace=0.2)

# Optional: Add a title to the entire figure
# fig.suptitle('Distribution of Values Across Categories and Columns', fontsize=16, y=0.99)

# --- Display Plot ---
plt.show()