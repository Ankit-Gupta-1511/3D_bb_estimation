import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import VGG16
import keras.backend as K

number_bin = 2

def build_model():
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable=False
    x = base_model.get_layer('block5_pool').output
    x = tf.keras.layers.Flatten()(x)
    # dimesion head
    dimension = tf.keras.layers.Dense(512)(x)
    dimension = tf.keras.layers.LeakyReLU(alpha=0.1)(dimension)
    dimension = tf.keras.layers.Dropout(0.5)(dimension)
    dimension = tf.keras.layers.Dense(3)(dimension)
    dimension = tf.keras.layers.LeakyReLU(alpha=0.1, name='dimension')(dimension)

    # orientation head
    orientation = tf.keras.layers.Dense(256)(x)
    orientation = tf.keras.layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = tf.keras.layers.Dropout(0.5)(orientation)
    orientation = tf.keras.layers.Dense(number_bin*2)(orientation)
    orientation = tf.keras.layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = tf.keras.layers.Reshape((number_bin,-1))(orientation)
    orientation = tf.keras.layers.Lambda(K.l2_normalize, name='orientation')(orientation)

    # confidence head
    confidence = tf.keras.layers.Dense(256)(x)
    confidence = tf.keras.layers.LeakyReLU(alpha=0.1)(confidence)
    confidence = tf.keras.layers.Dropout(0.5)(confidence)
    confidence = tf.keras.layers.Dense(number_bin, activation='softmax', name='confidence')(confidence)
    # model
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=[dimension, orientation, confidence])
    return model