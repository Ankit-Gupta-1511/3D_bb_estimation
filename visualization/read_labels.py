import os
import numpy as np

def read_labels(label_dir, img_idx):
    # Get the list of files in the label directory and sort them
    file_list = sorted(os.listdir(label_dir))
    
    # Read the label file for the specified image index
    # Adjust the index by 3 based on the sorting order
    label_file = file_list[3 + img_idx]
    
    # Open the label file and parse the contents
    objects = []
    with open(os.path.join(label_dir, label_file), 'r') as file:
        for line in file:
            # Split the line into tokens
            tokens = line.split()
            object_data = {
                'type': tokens[0],  # 'Car', 'Pedestrian', ...
                'truncation': float(tokens[1]),  # Truncated pixel ratio ([0..1])
                'occlusion': int(tokens[2]),  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
                'alpha': float(tokens[3]),  # Object observation angle ([-pi..pi])
                'x1': float(tokens[4]),  # Left
                'y1': float(tokens[5]),  # Top
                'x2': float(tokens[6]),  # Right
                'y2': float(tokens[7]),  # Bottom
                'h': float(tokens[8]),   # Box width
                'w': float(tokens[9]),   # Box height
                'l': float(tokens[10]),  # Box length
                't': [float(tokens[11]), float(tokens[12]), float(tokens[13])],  # Location (x, y, z)
                'ry': float(tokens[14])  # Yaw angle
            }
            # Adjust the 'ry' value based on location and yaw angle
            object_data['ry'] = object_data['ry'] + np.arctan2(object_data['t'][0], object_data['t'][2])
            
            objects.append(object_data)
    
    return objects

# Example usage:
# label_dir = '/path/to/label/files'
# img_idx = 0  # Index of the image
# objects = read_labels(label_dir, img_idx)
