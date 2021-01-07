import cv2
import numpy as np
import pickle

# get labels instead of indices
rev_lables = {}
with open("ids.pickle",'rb') as file:
    labels = pickle.load(file)
    # it would normally be {name, index}
    rev_labels = {value: key for key, value in labels.items()}

face_cascade = cv2.CascadeClassifier('cascades/frontface.xml')
img = cv2.imread('images/test.PNG') # path to my test image
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

recognized = False

while True:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        roi = gray[y:y+h, x:x+w]

        # recognizing
        _id_, confidence = recognizer.predict(roi)
        if confidence <= 45:
            if not recognized:
                print(confidence)
                print(rev_labels[_id_])
                recognized = True

                color = (0, 255, 0)  # BGR
                stroke = 2
                end_coord_x = x + w
                end_coord_y = y + h
                cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)
        else:
            color = (0, 0, 255)  # BGR
            stroke = 2
            end_coord_x = x + w
            end_coord_y = y + h
            cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # Display frame
    cv2.imshow('frame', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()