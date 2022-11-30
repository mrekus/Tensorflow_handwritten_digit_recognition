import tensorflow as tf
import cv2

# Užkraunamas mnist dataset iš tensorflow
mnist = tf.keras.datasets.mnist

# Padalinami duomenys į train ir test datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizuojam duomenis, iš (0 - 255 juodų, baltų)(dalinama iš 255)
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

