from yolo_v4_tf.models import Yolov4
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import lib
import numpy as np
import os 
from visualize import render_predictions

# Assuming 'output' is the name of the folder you want to save your images in.
output_folder = '/home/ankit/work/3D_bb_estimation/validation'

image_folder = '/home/ankit/work/3D_bb_estimation/kitti/data_object_image_2/testing/image_2'
yolo_weight_path = '/home/ankit/work/3D_bb_estimation/model/weights_config/yolov4.weights'
coco_classes = '/home/ankit/work/3D_bb_estimation/model/weights_config/coco_classes.txt'
weight_path_3d = '/home/ankit/work/3D_bb_estimation/model/weights_config/weights.hdf5'

calibration_folder = '/home/ankit/work/3D_bb_estimation/kitti/data_object_calib/testing/calib'
label_folder = '/home/ankit/work/3D_bb_estimation/kitti/data_object_label_2/testing/label_2'
image_size = (224, 224)
number_bin = 2

def predict(image_path):
    # load yolo model and get 2d detection
    model = Yolov4(weight_path=yolo_weight_path, class_name_path=coco_classes)
    detection_2d = model.predict(image_path)

    # load 3d bb estimation model
    estimation_model_3d = lib.build_model()
    estimation_model_3d.load_weights(weight_path_3d)

    # read image
    image_array = cv2.imread(image_path)

    # average dimension of car calculated using trainning data
    average_dims = np.array([1.52131309, 1.64441358, 3.85728004])

    # estimate 3d bb for each 2d detection
    predictions = []
    for i, row in detection_2d.iterrows():
        # crop image using detection
        cropped_image = image_array[int(row["y1"]):int(row["y2"]),
                               int(row["x1"]):int(row["x2"])]
        cropped_image = cv2.resize(cropped_image, image_size)
        cropped_image = np.expand_dims(cropped_image, axis=0)
        # estimate 3d
        cropped_image = cropped_image/255
        prediction_3d = estimation_model_3d.predict(cropped_image)
        # dimension
        # need to add average dimension to prediction
        # minus done for training
        dimension = prediction_3d[0][0]
        dimension = dimension+average_dims
        # get angle
        confidence = np.argmax(prediction_3d[2][0])
        anchors = prediction_3d[1][0][confidence]
        if anchors[1] > 0:
            angle_offset = np.arccos(anchors[0])
        else:
            angle_offset = -np.arccos(anchors[0])
        wedge = 2.*np.pi/number_bin
        angle_offset = angle_offset + confidence*wedge
        angle_offset = angle_offset % (2.*np.pi)
        angle_offset = angle_offset - np.pi/2
        if angle_offset > np.pi:
            angle_offset = angle_offset - (2.*np.pi)
        # collect all detections
        detections = {
            "x1": int(row["x1"]),
            "y1": int(row["y1"]),
            "x2": int(row["x2"]),
            "y2": int(row["y2"]),
            "dimension": dimension,
            "angle_offset": angle_offset,
            "type": row["class_name"]
        }
        predictions.append(detections)
    return predictions

# Load image & run testing
all_image = [f for f in sorted(os.listdir(image_folder)) if f.endswith('.png')]
calibration_paths = [f for f in sorted(os.listdir(calibration_folder)) if f.endswith('.txt')]
# label_paths = [f for f in sorted(os.listdir(label_folder)) if f.endswith('.txt')]


for image_idx, f in enumerate(all_image):
    image_file = image_folder +'/' + f
    calibration_path = calibration_folder + '/' + calibration_paths[image_idx]
    # label_path = label_folder + '/' + label_paths[image_idx]
    print(image_file)
    print(calibration_path)
    predictions = predict(image_file)
    render_predictions(predictions, image_file, output_folder, calibration_path, '')