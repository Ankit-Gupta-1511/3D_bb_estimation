import numpy as np

def calc_theta_ray(img, prediction, camera_calib):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * camera_calib[0][0]))
        center = (prediction['x1'] + prediction['x2']) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle


def calc_ry(img, angle, prediction, camera_calib):
    return angle + calc_theta_ray(img, prediction, camera_calib)
