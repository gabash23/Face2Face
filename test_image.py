import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades/frontface.xml')
img = cv2.imread('images/Emilio/3.jpg') # path to my test image

while True:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        color = (255, 0, 0)  # BGR
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # Display frame
    cv2.imshow('frame', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()