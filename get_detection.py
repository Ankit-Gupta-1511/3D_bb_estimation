from yolo_v4_tf.models import Yolov4
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import lib
import numpy as np

image_path = 'path_to_image/training/image_2/000024.png'
yolo_weight_path = 'path_to_weights_config/weights_config/yolov4.weights'
coco_classes = 'path_to_weights_config/weights_config/coco_classes.txt'
weight_path_3d = 'path_to_weights_config/weights_config/weights.hdf5'
image_size = (224, 224)
number_bin = 2

def predict():
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
            "angle_offset": angle_offset
        }
        predictions.append(detections)
    print(predictions)
    return predictions
predict()