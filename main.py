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
# predictions = model.predict([X_testr])  # Didžiausia tikimybė bus atsakymas
# Predictions turi visą dataset, su np.argmax(predictions[xxx]) galima rasti kiekvieno dataset
# elemento prediction pagal indeksą


# Prediction ranka parašytam skaičiui
# Užkraunam baltą skaičių juodam fone
img = cv2.imread("4.png")

# pakeičiamas į pilką
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize
resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

# Normalizavimas
newimg = tf.keras.utils.normalize(resized, axis=1)  # Normalizuoja tarp 0 ir 1 (dalyba iš 255)
newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Pridedama dimensija

# Spėjamas skaičius su sukurtu modeliu
predictions = model.predict(newimg)
print(f"Atsakymas: {np.argmax(predictions)}")  # Spėjimo atsakymas

# Tiesiogiai per kamerą
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise OSError("Cannot open camera")


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1)
    newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    predictions = model.predict(newimg)
    status = np.argmax(predictions)

    x, y, w, h = 0, 0, 100, 100
    cv2.rectangle(frame, (x, x), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, status.astype(str), (x + int(w/5), y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Camera feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cap.release()
    cv2.destroyAllWindows()
