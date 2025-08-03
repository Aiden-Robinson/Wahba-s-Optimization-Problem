import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define reference vectors (world frame)
v1 = np.array([0, 0, -1])  # gravity
v2 = np.array([1, 0, 0])   # magnetic field
origin = np.array([0, 0, 0])

# Apply ground-truth rotation
r_gt = R.from_euler('xyz', [30, -20, 45], degrees=True)
R_gt = r_gt.as_matrix()  # the rotation matrix

# Rotate vectors to get body-frame measurements
u1 = R_gt @ v1
u2 = R_gt @ v2

# Step 2: Add Noise
# Simulate realistic sensor noise with Gaussian disturbance
def add_noise(v: np.ndarray, std: float = 0.01) -> np.ndarray:
    return v + np.random.normal(0, std, size=v.shape)

u1_noisy = add_noise(u1)
u2_noisy = add_noise(u2)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_title("Reference (World) vs Rotated (Body) Vectors (With Noise)")

# Plot reference vectors (world frame)
ax.quiver(*origin, *v1, color='blue', label='v1 (gravity)', linewidth=2)
ax.quiver(*origin, *v2, color='green', label='v2 (magnetic)', linewidth=2)

# Plot rotated vectors (body frame, clean)
ax.quiver(*origin, *u1, color='cyan', linestyle='dashed', label='u1 = R * v1 (clean)')
ax.quiver(*origin, *u2, color='lime', linestyle='dashed', label='u2 = R * v2 (clean)')

# Plot rotated vectors (body frame, noisy)
ax.quiver(*origin, *u1_noisy, color='magenta', linestyle='dotted', label='u1 (noisy)')
ax.quiver(*origin, *u2_noisy, color='orange', linestyle='dotted', label='u2 (noisy)')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
