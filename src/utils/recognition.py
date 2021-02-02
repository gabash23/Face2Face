from typing import Tuple, List, Union

import cv2
import numpy as np
import os
import pickle


def recognize(filePath) -> Union[str, Tuple[str, int]]:
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
    ret_str: str = " "
    conf: int = -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        color_roi = img[y:y + h, x:x + w]

        # recognizing
        _id_, confidence = recognizer.predict(roi)
        if confidence <= 45:
            color = (0, 0, 255)  # BGR
            stroke = 8
            end_coord_x = x + w
            end_coord_y = y + h

            if not recognized:
                ret_str = labels[_id_]
                conf = confidence
                recognized = True
                path = 'src/static'
                cv2.imwrite(os.path.join(path, labels[_id_].title() + ".png"),
                            cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke))

    if ret_str == " ":
        return "Unknown"

    ret_str = ret_str.replace("_", " ").title()

    return ret_str, conf
