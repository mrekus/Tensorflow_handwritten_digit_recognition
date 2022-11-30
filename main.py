import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import cv2
import numpy as np

# Užkraunamas mnist dataset iš tensorflow
mnist = tf.keras.datasets.mnist

# Padalinami duomenys į train ir test datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizuojam duomenis, iš (0 - 255 juodų, baltų)(dalinama iš 255)
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

IMG_SIZE = 28
# Pridedama viena dimensija norint atlikti Convolution, -1 reiškia didžiausią reikšmę
X_trainr = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_testr = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Kuriamas deep learning network
model = Sequential()

# Pirmas Convolution sluoksnis (filtrų (kernels) skaičius 64, dydis 3x3) (28 - 3 + 1) = 26 x 26
model.add(
    Conv2D(64, (3, 3), input_shape=X_trainr.shape[1:])
)  # Imam nuo 1, kad paimtume vieno dydį, ne skaičių (60000, 28, 28, 1)
model.add(
    Activation("relu")
)  # Aktyvacijos f-ja padaro non-linear (išmeta visas <0 reikšmes, visas >0 praleidžia į kitą sluoksnį
model.add(
    MaxPool2D(pool_size=(2, 2))
)  # Praleis tik didžiausią 2x2 matricos reikšmę, kitas išmes

# Antras Convolution sluoksnis (26 - 3 + 1) = 24 x 24
model.add(Conv2D(64, (3, 3), input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

# Trečias Convolution sluoksnis (24 - 3 + 1) = 22 x 22
model.add(Conv2D(64, (3, 3), input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

# Pilnai sujungtas sluoksnis #1
model.add(Flatten())  # ištiesina pvz 22x22=484
model.add(Dense(64))  # neural network sluoksnis (visi neurons sujungti) (pvz visi 484 sujungti su visais 64)
model.add(Activation("relu"))

