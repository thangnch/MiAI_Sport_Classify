from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os



LABELS = set(["boxing", "swimming", "football"])

# Load image de xu ly anh
imagePaths = list(paths.list_images("data"))
data = []
labels = []
# Loop thu muc data

for imagePath in imagePaths:

	label = imagePath.split(os.path.sep)[-2]

	# Xu ly anh
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# Them vao du lieu data va label
	data.append(image)
	labels.append(label)


# Chuyen thanh numpy array va one hot
data = np.array(data)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Chia train/val theo ty le 80/20
(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.2, stratify=labels, random_state=42)

# Augment anh input
trainAug = ImageDataGenerator(	rotation_range=30,	zoom_range=0.15,	width_shift_range=0.2,	height_shift_range=0.2,	shear_range=0.15,	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# Su dung mang ResNet50 voi weight imagenet
baseModel = ResNet50(weights="imagenet", include_top=False,	input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# Dong bang cac lop cua mang Resnet
for layer in baseModel.layers:
	layer.trainable = False

opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])

# Train model
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=32),	steps_per_epoch=len(trainX) // 32,	validation_data=valAug.flow(testX, testY),	validation_steps=len(testX) // 32,
	epochs=25)

# Luu model
model.save("models/sport.h5")

# Luu ten class
f = open("models/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()


