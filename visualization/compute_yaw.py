import numpy as np

def calc_theta_ray(img, prediction, camera_calib):
        center = (prediction['x1'] + prediction['x2']) / 2
        u_distance = center - camera_calib[0, 2]
        focal_length = camera_calib[0, 0]
        angle = np.arctan(u_distance / focal_length)

        return angle


def calc_ry(img, angle, prediction, camera_calib):
    return angle + calc_theta_ray(img, prediction, camera_calib)
