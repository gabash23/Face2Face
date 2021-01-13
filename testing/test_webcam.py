import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/frontface.xml')
cap = cv2.VideoCapture(0)

while True:

    # Capture frame by frame
    ret, frame = cap.read()
    frame_flip = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        color = (255, 0, 0)  # BGR
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame_flip, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # Display frame
    cv2.imshow('frame', frame_flip)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
