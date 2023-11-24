import os
import cv2
import numpy as np
import itertools

from visualization.calibration import read_calibration_single
from visualization.read_labels import read_labels
from visualization.compute_box_3d import compute_box_3d
from visualization.compute_orientation_3d import compute_orientation_3d

def render_predictions(predictions, image_path, output_folder, calibration_path, label_dir):
    # Read the original image
    image = cv2.imread(image_path)

    P = read_calibration_single(calibration_path, cam=2) # cam = 2 means left color camera

    print(P)

    objects = read_labels(label_dir, img_idx=25)
    print(objects)

    for object_ in objects:
        # Plot 3D bounding box
        corners_2D, _ = compute_box_3d(object_, P)
        orientation_2D = compute_orientation_3d(object_, P)
        print(corners_2D)
        print(orientation_2D)

        # Draw 3D bounding box if corners are available
        if corners_2D.size != 0:
            color = (255, 0, 0)
            thickness = 2
            for i in range(4):
                # Draw lines between the base corners
                cv2.line(image, (int(corners_2D[0, i]), int(corners_2D[1, i])),
                        (int(corners_2D[0, (i+1) % 4]), int(corners_2D[1, (i+1) % 4])), color, thickness)
                
                # Draw lines between the top corners
                cv2.line(image, (int(corners_2D[0, i+4]), int(corners_2D[1, i+4])),
                        (int(corners_2D[0, (i+1) % 4 + 4]), int(corners_2D[1, (i+1) % 4 + 4])), color, thickness)
                
                # Draw lines between the base and top corners
                cv2.line(image, (int(corners_2D[0, i]), int(corners_2D[1, i])),
                        (int(corners_2D[0, i+4]), int(corners_2D[1, i+4])), color, thickness)

        # Draw orientation vector if available
        if orientation_2D.size != 0:
            cv2.line(image,
                     (int(orientation_2D[0, 0]), int(orientation_2D[1, 0])),
                     (int(orientation_2D[0, 1]), int(orientation_2D[1, 1])),
                     (0, 255, 255), 2)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with drawn 3D bounding boxes
    output_file_path = os.path.join(output_folder, 'output_image.png')
    cv2.imwrite(output_file_path, image)

    print(f"Image saved to {output_file_path}")

