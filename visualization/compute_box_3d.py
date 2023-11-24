import numpy as np

def project_to_image(points_3D, P):
    # Convert to homogeneous coordinates
    points_3D_hom = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))
    # Project to image plane
    points_2D_hom = np.dot(P, points_3D_hom)
    # Convert back to 2D
    points_2D = points_2D_hom[:2, :] / points_2D_hom[2, :]
    return points_2D

def compute_box_3d(_object, P):
    # Define the face index matrix
    face_idx = np.array([
        [0, 1, 5, 4],  # Front face
        [1, 2, 6, 5],  # Left face
        [2, 3, 7, 6],  # Back face
        [3, 0, 4, 7],  # Right face
    ])

    # Compute the rotation matrix around the yaw axis
    R = np.array([
        [np.cos(_object['ry']), 0, np.sin(_object['ry'])],
        [0, 1, 0],
        [-np.sin(_object['ry']), 0, np.cos(_object['ry'])]
    ])

    # 3D bounding box dimensions
    l = _object['l']
    w = _object['w']
    h = _object['h']

    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # Rotate and translate the 3D bounding box
    corners_3D = R @ np.array([x_corners, y_corners, z_corners])
    corners_3D += np.array(_object['t']).reshape(3, 1)

    # Only draw 3D bounding box for objects in front of the camera
    if np.any(corners_3D[2, :] < 0.1):
        corners_2D = np.array([])
        return corners_2D, face_idx

    # Project the 3D bounding box into the image plane
    corners_2D = project_to_image(corners_3D, P)

    return corners_2D, face_idx

# Example usage:
# _object = {
#     'ry': np.deg2rad(45),  # Rotation around the yaw axis in radians
#     'l': 4.0,              # Length of the car
#     'w': 1.8,              # Width of the car
#     'h': 1.6,              # Height of the car
#     't': [1.0, 0.5, 20.0]  # Translation vector (x, y, z) in meters
# }
# P = np.array([...])  # Your 3x4 projection matrix

# corners_2D, face_idx = compute_box_3d(_object, P)
