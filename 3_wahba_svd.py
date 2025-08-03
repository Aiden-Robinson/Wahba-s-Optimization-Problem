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

def add_noise(v: np.ndarray, std: float = 0.01) -> np.ndarray:
    return v + np.random.normal(0, std, size=v.shape)

u1_noisy = add_noise(u1)
u2_noisy = add_noise(u2)

# Step 3: Solve Wahba’s Problem (Closed-form with SVD)
# Construct matrix B = v1 * u1_noisy^T + v2 * u2_noisy^T
A = np.outer(v1, u1_noisy) + np.outer(v2, u2_noisy)
U, _, Vt = np.linalg.svd(A)
R_est = U @ Vt

print("Estimated Rotation Matrix (R_est):\n", R_est)
print("Ground Truth Rotation Matrix (R_gt):\n", R_gt)

# Optionally, visualize the estimated rotation
u1_est = R_est @ v1
u2_est = R_est @ v2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_title("Wahba's Problem: Estimated vs Ground Truth Vectors")

# Plot reference vectors (world frame)
ax.quiver(*origin, *v1, color='blue', label='v1 (gravity)', linewidth=2)
ax.quiver(*origin, *v2, color='green', label='v2 (magnetic)', linewidth=2)

# Plot noisy body-frame measurements
ax.quiver(*origin, *u1_noisy, color='magenta', linestyle='dotted', label='u1 (noisy)')
ax.quiver(*origin, *u2_noisy, color='orange', linestyle='dotted', label='u2 (noisy)')

# Plot estimated rotated vectors
ax.quiver(*origin, *u1_est, color='red', linestyle='dashed', label='u1 (est)')
ax.quiver(*origin, *u2_est, color='yellow', linestyle='dashed', label='u2 (est)')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
