import numpy as np
import cv2
import pickle


width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

emoji_types = ['Happy', 'Sad', 'Angry', 'Poo', 'Surprised'] #["Angry", "Happy", "Poo", "Sad", "Surprised"]
while True:
    success, img_original = cap.read()
    img = np.asarray(img_original)
    img = cv2.resize(img, (32, 32))
    img = pre_process(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # Predict class
    emoji_class_index = int(model.predict_classes(img))
    predictions = model.predict(img)
    probability_value = np.amax(predictions)
    print(emoji_types[emoji_class_index], probability_value)

    if probability_value > 0.8:
        cv2.putText(img_original, emoji_types[emoji_class_index] + "   "+ str(probability_value),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
    cv2.imshow("Original Image", img_original)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        break