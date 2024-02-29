import numpy as np
import matplotlib.pyplot as plt

# Define weights and bias
w1, w2, w0 = 1, -1, 5  # Example values

# Create a grid of x1 and x2 values
x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

# Compute w1*x1 + w2*x2 + w0
z = w1 * x1 + w2 * x2 + w0

# Number of points
num_points = 10

# Generate random points above and below the plane
above_plane_x1 = np.random.uniform(-10, 10, num_points)
above_plane_x2 = (w1 * above_plane_x1 + w0) / -w2 + np.random.uniform(0, 5, num_points)  # Ensure they are above the plane

below_plane_x1 = np.random.uniform(-10, 10, num_points)
below_plane_x2 = (w1 * below_plane_x1 + w0) / -w2 - np.random.uniform(0, 5, num_points)  # Ensure they are below the plane

# Plot
plt.figure(figsize=(6, 6))

# Plot the plane
plt.contourf(x1, x2, z, levels=[0, z.max()], alpha=0.3, colors=['blue'])

# Plot green points above the plane
plt.scatter(above_plane_x1, above_plane_x2, color='green', marker='o', edgecolors='black', label='Above Plane (Green)')

# Plot red points below the plane
plt.scatter(below_plane_x1, below_plane_x2, color='red', marker='o', edgecolors='black', label='Below Plane (Red)')

# Add details
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title(f'Plane with $w_1x_1 + w_2x_2 + w_0 \geq 0$ \n($w_1={w1}$, $w_2={w2}$, $w_0={w0}$)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.ylim(-10,10)
plt.grid(True)
plt.show()
