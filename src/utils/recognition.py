from typing import Tuple, List

import cv2
import numpy as np
import pickle


def recognize(filePath):
    # get labels instead of indices
    with open("src/utils/ids.pickle", 'rb') as file:
        temp = pickle.load(file)
        # it would normally be {name, index}
        labels = {value: key for key, value in temp.items()}

    face_cascade = cv2.CascadeClassifier('src/utils/cascades/frontface.xml')
    img = cv2.imread(filePath)  # path to my test image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("src/utils/trainer.yml")

    recognized: bool = False
    recognized_list: List[str] = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]

        # recognizing
        _id_, confidence = recognizer.predict(roi)
        if confidence <= 45:
            if not recognized:
                recognized_list.append(labels[_id_])
                recognized = True

    if not len(recognized_list):
        return "Unknown"

    ret_str = ""
    for element in recognized_list:
        ret_str += element.replace("_", " ").title() + " "
    return ret_str
