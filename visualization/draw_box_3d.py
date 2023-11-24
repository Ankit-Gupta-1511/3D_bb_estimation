import matplotlib.pyplot as plt
import numpy as np

def draw_box_3d(object, corners, face_idx, orientation):
    # Set styles for occlusion and truncation
    occ_col = ['g', 'y', 'r', 'w']
    trun_style = ['-', '--']
    trc = int(object['truncation'] > 0.1) + 1

    fig, ax = plt.subplots()
    
    # Draw projected 3D bounding boxes
    if corners.size != 0:
        for f in range(4):
            face = face_idx[f]
            x = [corners[0, idx] for idx in face] + [corners[0, face[0]]]
            y = [corners[1, idx] for idx in face] + [corners[1, face[0]]]
            ax.plot(x, y, color=occ_col[object['occlusion']], linewidth=3, linestyle=trun_style[trc])
            ax.plot(x, y, color='b', linewidth=1)

    # Draw orientation vector
    if orientation.size != 0:
        ax.plot([orientation[0, :], orientation[0, :]],
                [orientation[1, :], orientation[1, :]],
                color='w', linewidth=4)
        ax.plot([orientation[0, :], orientation[0, :]],
                [orientation[1, :], orientation[1, :]],
                color='k', linewidth=2)

    plt.show()

# Example usage
# object_example = {'truncation': 0.5, 'occlusion': 2}
# corners_example = np.array([[100, 200, 200, 100], [100, 100, 200, 200]])  # Example corner points
# face_idx_example = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]  # Example face indices
# orientation_example = np.array([[150, 150], [125, 175]])  # Example orientation vector

# draw_box_3d(object_example, corners_example, face_idx_example, orientation_example)
