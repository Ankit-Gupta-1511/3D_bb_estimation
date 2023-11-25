import os
import cv2
import numpy as np
import itertools

from visualization.calibration import read_calibration_single
from visualization.read_labels import read_labels
from visualization.compute_box_3d import compute_box_3d
from visualization.compute_orientation_3d import compute_orientation_3d
from visualization.compute_translation import compute_translation
from visualization.compute_yaw import calc_ry


def convert_predictions(image, predictions, camera_calib):
    objects = []
    for prediction in predictions:
        angle = prediction['angle_offset']
        ry = calc_ry(image, angle, prediction, camera_calib)

        obj = {
                'type': prediction['type'],  # 'Car', 'Pedestrian', ...
                'alpha': prediction['angle_offset'],  # Object observation angle ([-pi..pi])
                'x1': prediction['x1'],  # Left
                'y1': prediction['y1'],  # Top
                'x2': prediction['x2'],  # Right
                'y2': prediction['y2'],  # Bottom
                'h': prediction['dimension'][0],   # Box width
                'w': prediction['dimension'][1],   # Box height
                'l': prediction['dimension'][2],  # Box length
                't': [],  # Location (x, y, z)
                'ry': ry  # Yaw angle
        }

        t = compute_translation(camera_calib, obj, angle, ry)
        print('t = ', t)
        obj['t'] = t

        objects.append(obj)
    return objects


def render_predictions(predictions, image_path, output_folder, calibration_path, label_dir):
    # Read the original image
    image = cv2.imread(image_path)

    P = read_calibration_single(calibration_path, cam=2) # cam = 2 means left color camera

    lable_objects = read_labels(label_dir, img_idx=24)
    label_translation = compute_translation(P, lable_objects[0], lable_objects[0]['alpha'], lable_objects[0]['ry'])
    # print("label translation, ", label_translation)
    label_ry = calc_ry(image, lable_objects[0]['alpha'], lable_objects[0], P)
    print("label ry, ", label_ry)
    objects = convert_predictions(image, predictions, P)
    # print("Predictions")
    # print(predictions)
    print("Objects from given labels")
    print(lable_objects)
    print("computed objects")
    print(objects)

    for object_ in objects:
        # Plot 3D bounding box
        corners_2D, face_idx = compute_box_3d(object_, P)
        orientation_2D = compute_orientation_3d(object_, P)
        # print(corners_2D)
        # print(orientation_2D)

        # Draw 3D bounding box if corners are available
        thickness = 1
        if corners_2D.size != 0:
            color = (255, 0, 0)
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
                        
            # Draw cross on the front face
            front_face_corners = corners_2D[:, face_idx[0]]  # Get the front face corners
            # Draw lines between opposite corners to form the cross
            cv2.line(image, (int(front_face_corners[0, 0]), int(front_face_corners[1, 0])),
                    (int(front_face_corners[0, 2]), int(front_face_corners[1, 2])), (0, 0, 255), thickness)
            cv2.line(image, (int(front_face_corners[0, 1]), int(front_face_corners[1, 1])),
                    (int(front_face_corners[0, 3]), int(front_face_corners[1, 3])), (0, 0, 255), thickness)
                     
        # Draw orientation vector if available
        if orientation_2D.size != 0:
            cv2.line(image,
                     (int(orientation_2D[0, 0]), int(orientation_2D[1, 0])),
                     (int(orientation_2D[0, 1]), int(orientation_2D[1, 1])),
                     (0, 255, 255), thickness)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with drawn 3D bounding boxes
    output_file_path = os.path.join(output_folder, 'output_image.png')
    cv2.imwrite(output_file_path, image)

    print(f"Image saved to {output_file_path}")

