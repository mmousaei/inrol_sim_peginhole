import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def interpolate_angles(start_angle, end_angle, steps):
    """
    Interpolates from start_angle to end_angle (in degrees) over 'steps' increments,
    handling wrap-around at the -180/180 boundary.
    """
    # Normalize angles to the range [-180, 180)
    start_angle = (start_angle + 180) % 360 - 180
    end_angle = (end_angle + 180) % 360 - 180

    # Calculate the shortest direction to interpolate
    delta = (end_angle - start_angle) % 360
    # Adjust delta for wrap-around
    if delta > 180:
        delta -= 360  # Going the shorter path backwards
    elif delta < -180:
        delta += 360  # Going the shorter path forwards

    # Interpolate without wrapping each step
    interpolated = [start_angle + (delta * step / steps) for step in range(steps)]

    # Normalize the result to ensure all angles are within [-180, 180)
    normalized = [(angle + 180) % 360 - 180 for angle in interpolated]

    return normalized

# Inputs
des_pos = np.array([0.521, 0, 0.07])  # Desired 3D position
des_roll = 179.99 
total_points = 1000
total_time = 5  # Total time to complete the motion in seconds
start_time_gripper_turn = 0.5 * total_time  # Time to start gripper turning
end_time_gripper_turn = total_time  # Time to end gripper turning

# Time vector
time = np.linspace(0, total_time, total_points)

# Calculate indices for start and end of gripper turn within the total_points
start_idx_gripper_turn = int(total_points * (start_time_gripper_turn / total_time))

# Position interpolation (linear) - Adjusted for early completion
initial_pos = np.array([0.521, 0, 0.2])  # Initial position
pd = np.array([initial_pos + (des_pos - initial_pos) * (i / start_idx_gripper_turn) for i in range(start_idx_gripper_turn)])
pd = np.vstack((pd, np.repeat(des_pos[np.newaxis, :], total_points - start_idx_gripper_turn, axis=0)))  # Maintain final position

# Rotation interpolation (SLERP)
initial_roll = 179.99
# Adjust for shortest path
rolls = interpolate_angles(initial_roll, des_roll, start_idx_gripper_turn)
pitchs = [0] * start_idx_gripper_turn
yaws = [0] * start_idx_gripper_turn

rots_until_start = R.from_euler('xyz', np.column_stack((rolls, pitchs, yaws)), degrees=True)

# Extend the final orientation across the remaining timeline
final_quat = rots_until_start.as_quat()[-1]
final_rots = np.tile(final_quat, (total_points - start_idx_gripper_turn, 1))
rots = R.from_quat(np.vstack((rots_until_start.as_quat(), final_rots)))

# Extract rotation matrices for Rd
Rd = np.array([rot.as_matrix() for rot in rots])

# Save to files
np.savetxt('pd.txt', pd.reshape(total_points, -1), fmt='%.6f')  # Transpose to match the requested format
np.savetxt('Rd.txt', Rd.reshape(total_points*3, -1), fmt='%.6f')  # Flatten each rotation matrix and transpose

# Print shapes as a placeholder for actual file saving
print("Shapes: pd {}, Rd {}".format(pd.shape, Rd.shape))
