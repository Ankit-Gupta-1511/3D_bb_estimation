from yolo_v4_tf.models import Yolov4
model = Yolov4(weight_path='/Users/privy/Downloads/yolov4.weights',
               class_name_path='/Users/privy/Downloads/coco_classes.txt')

prediction = model.predict('/Users/privy/Desktop/all_repos/3D_bb_estimation/data/training/image_2/000024.png')
print(prediction.head())