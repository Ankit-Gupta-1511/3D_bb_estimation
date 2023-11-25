import numpy as np
from scipy.linalg import svd
# Reference - https://cs.gmu.edu/~amousavi/papers/3D-Deepbox-Supplementary.pdf

def box3d_candidate(object, angle, soft_range):
    x_corners = [object['l'], object['l'], object['l'], object['l'], 0, 0, 0, 0]
    y_corners = [object['h'], 0, object['h'], 0, object['h'], 0, object['h'], 0]
    z_corners = [0, 0, object['w'], object['w'], object['w'], object['w'], 0, 0]

    x_corners = [i - object['l'] / 2 for i in x_corners]
    y_corners = [i - object['h'] for i in y_corners]
    z_corners = [i - object['w'] / 2 for i in z_corners]

    corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
    point1 = corners_3d[0, :]
    point2 = corners_3d[1, :]
    point3 = corners_3d[2, :]
    point4 = corners_3d[3, :]
    point5 = corners_3d[6, :]
    point6 = corners_3d[7, :]
    point7 = corners_3d[4, :]
    point8 = corners_3d[5, :]

    # set up projection relation based on local orientation
    xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

    if 0 < angle < np.pi / 2:
        xmin_candi = point8
        xmax_candi = point2
        ymin_candi = point2
        ymax_candi = point5

    if np.pi / 2 <= angle <= np.pi:
        xmin_candi = point6
        xmax_candi = point4
        ymin_candi = point4
        ymax_candi = point1

    if np.pi < angle <= 3 / 2 * np.pi:
        xmin_candi = point2
        xmax_candi = point8
        ymin_candi = point8
        ymax_candi = point1

    if 3 * np.pi / 2 <= angle <= 2 * np.pi:
        xmin_candi = point4
        xmax_candi = point6
        ymin_candi = point6
        ymax_candi = point5

    # soft constraint
    div = soft_range * np.pi / 180
    if 0 < angle < div or 2*np.pi-div < angle < 2*np.pi:
        xmin_candi = point8
        xmax_candi = point6
        ymin_candi = point6
        ymax_candi = point5

    if np.pi - div < angle < np.pi + div:
        xmin_candi = point2
        xmax_candi = point4
        ymin_candi = point8
        ymax_candi = point1

    return xmin_candi, xmax_candi, ymin_candi, ymax_candi


def compute_translation(camera_calib, object, alpha, ry):
    # Extract the 2D bounding box coordinates and the 3D dimensions from the object dictionary
    bbox = [object['x1'], object['y1'], object['x2'], object['y2']]
    dimensions = [object['l'], object['w'], object['h']]

    l = object['l']
    w = object['w']
    h = object['h']
    
    K = camera_calib

    # Construct the rotation matrix around the yaw axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # format 2d corners
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimensions[0] / 2
    dy = dimensions[1] / 2
    dz = dimensions[2] / 2

    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-1,1):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=float)
        b = np.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            M = np.dot(K, M)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc