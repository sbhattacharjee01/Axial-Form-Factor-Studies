import numpy as np
import matplotlib.pyplot as plt

def draw_pie(ax, sizes, colors, xy, pie_radius):
    """Draw a compact small pie chart at a given (x, y) location"""
    x, y = xy
    pie_ax = ax.inset_axes([x - pie_radius/2, y - pie_radius/2, pie_radius, pie_radius], transform=ax.transData)
    pie_ax.pie(sizes, colors=colors, radius=1)
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])
    pie_ax.set_frame_on(False)

# Define grid points and calculate proper spacing
n_pies = 10  # Number of pies along each axis
grid_size = 80  # Total size of the grid
spacing = grid_size / (n_pies - 1)  # Distance between grid points
pie_radius = spacing * 0.9  # Adjust radius to make pies touch but not overlap

x = np.linspace(-grid_size / 2, grid_size / 2, n_pies)
y = np.linspace(-grid_size / 2, grid_size / 2, n_pies)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-grid_size / 2 - spacing/2, grid_size / 2 + spacing/2)
ax.set_ylim(-grid_size / 2 - spacing/2, grid_size / 2 + spacing/2)
ax.set_xticks(x)
ax.set_yticks(y)
ax.grid(True, linestyle='--', alpha=0.6)

# Define colors and random data
colors = ['magenta', 'lime', 'yellow', 'gray']
np.random.seed(42)

for i in x:
    for j in y:
        sizes = np.random.rand(4)  # Random fractions for each category
        sizes /= np.sum(sizes)  # Normalize to sum to 1
        draw_pie(ax, sizes, colors, (i, j), pie_radius)

# Labels
ax.set_xlabel(r'$\delta\theta_P$')
ax.set_ylabel(r'$\delta\theta_R$')

plt.show()
