import numpy as np
import os

def read_calibration(calib_dir, img_idx, cam):
    file_list = sorted(os.listdir(calib_dir))
    
    # Load the 3x4 projection matrix for the specified image index and camera
    # Adjusting the index by 3 is specific to the dataset structure we're using 
    calib_file = file_list[3 + img_idx]
    
    # Read the calibration data from the file
    P = np.genfromtxt(os.path.join(calib_dir, calib_file), delimiter=' ')
    
    # Select the calibration for the specified camera and reshape the array
    P = P[cam, :].reshape(3, 4)
    
    return P


def read_calibration_single(calib_file_path, cam):
    # Read the content of the calibration file
    with open(calib_file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the specific camera projection matrix (P0, P1, P2, P3)
    P_line = lines[cam].strip().split(' ')[1:]  # Skip the "P0:" part
    P = np.array([float(value) for value in P_line]).reshape(3, 4)
    
    return P
    