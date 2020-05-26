import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

"""
Constants
"""
path = 'data'
test_ratio = 0.2
validation_ratio = 0.2
img_dim = 32

"""
Variables
"""
images = []
class_number = []

dir_list = os.listdir(path)

number_of_classes = len(dir_list)

"""
Import images and resize them.
"""
for x, folder in enumerate(dir_list):
    image_list = os.listdir(path + "/" + str(folder))
    for _, img in enumerate(image_list):
        current_img = cv2.imread(path + "/" + str(folder) + "/" + img)
        current_img = cv2.resize(current_img, (img_dim, img_dim))
        images.append(current_img)
        class_number.append(x)
    print(folder)
print(len(images))

images = np.array(images)
class_number = np.array(class_number)

print(images.shape)
print(class_number.shape)

"""
Split Data for training, testing, and validation.
"""
X_train, X_test, y_train, y_test = train_test_split(images, class_number, test_size = test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_ratio)
print(X_train.shape)
print(X_test.shape)

number_of_samples = []
for idx in range(number_of_classes):
    number_of_samples.append(len(np.where(y_train == idx)[0]))

print(number_of_samples)

plt.figure(figsize = (10, 5))
plt.bar(range(number_of_classes), number_of_samples)
plt.title("Number of images for each Emoji")
plt.xlabel("Emoji ID")
plt.ylabel("Number of images")
plt.show()


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

""" img = pre_process(X_train[20])
img = cv2.resize(img, (300, 300))
cv2.imshow("Preprocessed Image", img)
cv2.waitKey(0) """

X_train = np.array(list(map(pre_process, X_train)))
X_test = np.array(list(map(pre_process, X_test)))
X_validation = np.array(list(map(pre_process, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

data_generator = ImageDataGenerator(width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.2,
                                    shear_range = 0.1,
                                    rotation_range = 10)

data_generator.fit(X_train)

y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)
y_validation = to_categorical(y_validation, number_of_classes)

def create_model():
    number_of_filters = 60
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    number_of_nodes = 500

    model = Sequential()
    model.add(Conv2D(number_of_filters, size_of_filter1, input_shape = (img_dim, img_dim, 1), activation = "relu"))
    model.add(Conv2D(number_of_filters, size_of_filter1, activation = "relu"))
    model.add(MaxPooling2D(pool_size = size_of_pool))
    model.add(Conv2D(number_of_filters // 2, size_of_filter2, activation = "relu"))
    model.add(Conv2D(number_of_filters // 2, size_of_filter2, activation = "relu"))
    model.add(MaxPooling2D(pool_size = size_of_pool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(number_of_nodes, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation = "softmax"))
    model.compile(Adam(lr = 0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

model = create_model()
print(model.summary())