import numpy as np

# Reference - https://cs.gmu.edu/~amousavi/papers/3D-Deepbox-Supplementary.pdf

def compute_translation(camera_calib, prediction, rot_local, ry):
    bbox = [prediction['x1'], prediction['y1'], prediction['x2'], prediction['y2']]
    # rotation matrix
    R = np.array([[ np.cos(ry), 0,  np.sin(ry)],
                  [ 0,          1,  0         ],
                  [-np.sin(ry), 0,  np.cos(ry)]])
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    I = np.identity(3)

    xmin_candi, xmax_candi, ymin_candi, ymax_candi = obj.box3d_candidate(rot_local, soft_range=8)

    X  = np.bmat([xmin_candi, xmax_candi,
                  ymin_candi, ymax_candi])

    # X: [x, y, z] in object coordinate
    X = X.reshape(4,3).T

    # construct equation of the form AX = b
    for i in range(4):
        matrice = np.bmat([[I, np.matmul(R, X[:,i])], [np.zeros((1,3)), np.ones((1,1))]])
        M = np.matmul(camera_calib, matrice)

        if i % 2 == 0:
            A[i, :] = M[0, 0:3] - bbox[i] * M[2, 0:3]
            b[i, :] = M[2, 3] * bbox[i] - M[0, 3]

        else:
            A[i, :] = M[1, 0:3] - bbox[i] * M[2, 0:3]
            b[i, :] = M[2, 3] * bbox[i] - M[1, 3]

    # solve x, y, z, using method of least square
    # AX = b => X = inverse(A) x b
    Tran = np.matmul(np.linalg.pinv(A), b)

    tx, ty, tz = [float(np.around(tran, 2)) for tran in Tran]
    return tx, ty, tz