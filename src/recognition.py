import cv2
import numpy as np
import pickle


def recognize(filePath):
    # get labels instead of indices
    with open("utils/ids.pickle", 'rb') as file:
        temp = pickle.load(file)
        # it would normally be {name, index}
        labels = {value: key for key, value in temp.items()}

    face_cascade = cv2.CascadeClassifier('cascades/frontface.xml')
    img = cv2.imread(filePath)  # path to my test image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    recognized = False
    recognized_list = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        roi = gray[y:y + h, x:x + w]

        # recognizing
        _id_, confidence = recognizer.predict(roi)
        if confidence <= 45:
            if not recognized:
                recognized_list.append(labels[_id_])
                recognized = True

                # color = (0, 255, 0)  # BGR
                # stroke = 2
                # end_coord_x = x + w
                # end_coord_y = y + h
                # cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)
        # else:
        # color = (0, 0, 255)  # BGR
        # stroke = 2
        # end_coord_x = x + w
        # end_coord_y = y + h
        # cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)
        # print("Unknown... ")

    if not len(recognized_list):
        return "Unknown"

    ret_str = ""
    for element in recognized_list:
        ret_str = element + ", "
        return ret_str[:-2]


print(recognize("../images/test.jpg"))