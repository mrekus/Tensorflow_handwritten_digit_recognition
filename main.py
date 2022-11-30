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
)  # 64 convolution sluoksniai, Imam nuo 1, kad paimtume vieno dydį, ne skaičių (60000, 28, 28, 1)
model.add(
    Activation("relu")
)  # Aktyvacijos f-ja padaro non-linear (išmeta visas <0 reikšmes, visas >0 praleidžia į kitą sluoksnį
model.add(
    MaxPool2D(pool_size=(2, 2))
)  # Praleis tik didžiausią 2x2 matricos reikšmę, kitas išmes

# Antras Convolution sluoksnis
model.add(Conv2D(64, (3, 3), input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

# Trečias Convolution sluoksnis
model.add(Conv2D(64, (3, 3), input_shape=X_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

# Pilnai sujungtas sluoksnis #1
model.add(Flatten())  # ištiesina pvz 22x22=484
model.add(
    Dense(64)
)  # neural network sluoksnis (visi neurons sujungti) (pvz visi 484 sujungti su visais 64)
model.add(Activation("relu"))

# Pilnai sujungtas sluoksnis #2
model.add(Dense(32))
model.add(Activation("relu"))

# Paskutinis pilnai sujungtas sluoksnis, galutinis rezultatas turi būti lygus klasių skaičiui (šiuo atveju 10
# nes yra 10 skaičių nuo 0 iki 9)
model.add(Dense(10))
model.add(Activation("softmax"))  # Aktyvacijos f-ja softmax (Class probabilities)

# Jei būtų binary classification Dense layer būtų vienas neuronas ir paskutinė Activation f-ja būtų sigmoid
# Naudojam softmax, nes turim daug klasių

# print(model.summary())  # Atspausdina modelio suvestinę

# Modelio kompiliavimas
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)  # adam geriausias optimizer, fokusas yra accuracy

# Modelio treniravimas
model.fit(X_trainr, y_train, epochs=5, validation_split=0.3)

# Testavimas
test_loss, test_accuracy = model.evaluate(X_testr, y_test)

# Klasių tikimybės (softmax)
predictions = model.predict([X_testr])  # Didžiausia tikimybė bus atsakymas
# Predictions turi visą dataset, su np.argmax(predictions[xxx]) galima rasti kiekvieno dataset
# elemento prediction pagal indeksą
