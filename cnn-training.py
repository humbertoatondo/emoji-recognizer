import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

"""
Constants
"""
path = 'data'
test_ratio = 0.2
validation_ratio = 0.2

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
        current_img = cv2.resize(current_img, (32, 32))
        images.append(current_img)
        class_number.append(x)
    print(folder)
print(len(images))

images = np.array(images)
#images = images.reshape(images.shape[1:])
#images = images.transpose()
class_number = np.array(class_number)

print(images.shape)
print(class_number.shape)

"""
Split Data
"""
X_train, X_test, y_train, y_test = train_test_split(images, class_number, test_size = test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_ratio)
print(X_train.shape)
print(X_test.shape)