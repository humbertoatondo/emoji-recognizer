import numpy as np
import cv2
import os

path = 'data'

images = []
class_number = []

dir_list = os.listdir(path)

number_of_classes = len(dir_list)

"""
Import images and resize them.
"""
for _, folder in enumerate(dir_list):
    image_list = os.listdir(path + "/" + str(folder))
    class_number.append(folder)
    for _, img in enumerate(image_list):
        current_img = cv2.imread(path + "/" + str(folder) + "/" + img)
        current_img = cv2.resize(current_img, (32, 32))
        images.append(current_img)
    print(folder)
print(len(images))

images = np.array(images)
class_number = np.array(class_number)