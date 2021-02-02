import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('../src/utils/cascades/frontface.xml')
img = cv2.imread('../src/utils/images/gab.JPG') # path to my test image

while True:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    thresh = 127
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # for (x, y, w, h) in faces:
    #     # print(x, y, w, h)
    #     color = (255, 0, 0)  # BGR
    #     stroke = 2
    #     end_coord_x = x + w
    #     end_coord_y = y + h
    #     cv2.rectangle(img, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # Display frame
    cv2.imshow('frame', im_bw)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()