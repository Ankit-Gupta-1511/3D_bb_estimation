import numpy as np

def project_to_image(points_3D, P):
    # Convert to homogeneous coordinates
    points_3D_hom = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))
    # Project to image plane
    points_2D_hom = np.dot(P, points_3D_hom)
    # Convert back to 2D
    points_2D = points_2D_hom[:2, :] / points_2D_hom[2, :]
    return points_2D

def compute_orientation_3d(object_, P):
    # Compute rotational matrix around yaw axis
    R = np.array([
        [np.cos(object_['ry']), 0, np.sin(object_['ry'])],
        [0, 1, 0],
        [-np.sin(object_['ry']), 0, np.cos(object_['ry'])]
    ])

    # Orientation in object coordinate system (assuming the length of the arrow is the length of the object)
    orientation_3D = np.array([
        [0.0, object_['l']],
        [0.0, 0.0],
        [0.0, 0.0]
    ])

    # Rotate and translate in camera coordinate system
    orientation_3D = R @ orientation_3D
    orientation_3D += np.array(object_['t']).reshape(3, 1)

    # If vector is behind the image plane, return empty array
    if any(orientation_3D[2, :] < 0.1):
        return np.array([])

    # Project orientation into the image plane
    orientation_2D = project_to_image(orientation_3D, P)
    
    return orientation_2D

# Example usage:
# object_ = {
#     'ry': np.deg2rad(30),  # Yaw angle in radians
#     'l': 4.5,              # Length of the vehicle
#     't': [1.0, 2.0, 15.0]  # Translation vector of the vehicle
# }
# P = np.array([...])  # Replace with your 3x4 projection matrix

# orientation_2D = compute_orientation_3d(object_, P)
