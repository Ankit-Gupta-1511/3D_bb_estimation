import os
import cv2
import numpy as np
import itertools

def draw_box_3d(image, predictions, K):
    for detection in predictions:
        # Get the 2D bounding box coordinates
        x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        
        # Get the 3D dimensions (width, height, length)
        dimensions = detection['dimension']
        
        # Get the orientation angle
        angle = detection['angle_offset']

        # Compute the rotation matrix R from the yaw angle
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

        # Translation vector T is assumed to be the center of the bottom face of the 3D box
        T = np.array([(x1 + x2) / 2, y2, 0])  # Assuming the bottom center is aligned with the bottom of the 2D bbox

        # Get the 3D bounding box corners
        h, w, l = dimensions
        corners_3d = np.array([
            [l/2, w/2, 0], [l/2, -w/2, 0], [-l/2, w/2, 0], [-l/2, -w/2, 0],
            [l/2, w/2, -h], [l/2, -w/2, -h], [-l/2, w/2, -h], [-l/2, -w/2, -h]
        ])

        # Project the 3D corners to 2D
        corners_2d = project_to_image_plane(corners_3d, K, R, T)

        # Draw the edges of the cuboid
        for i, j in itertools.combinations(range(8), 2):
            # Check if the corners are on the same face of the cuboid
            if np.linalg.norm(corners_3d[i] - corners_3d[j]) in dimensions:
                cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[j].astype(int)), (0, 255, 0), 2)

    return image

def render_predictions(predictions, image_path, output_folder, K):
    # Read the original image
    image = cv2.imread(image_path)

    # Draw 3D boxes on the image
    image_with_boxes = draw_box_3d(image, predictions, K)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with drawn 3D bounding boxes
    output_file_path = os.path.join(output_folder, 'output_image.png')
    cv2.imwrite(output_file_path, image_with_boxes)

    print(f"Image saved to {output_file_path}")

