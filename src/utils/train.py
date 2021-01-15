import cv2
import numpy as np
import os
import pickle
from PIL import Image


def train_model():
    face_cascade = cv2.CascadeClassifier('src/utils/cascades/frontface.xml')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    curr_id = 0
    ids = {}
    x_train = []
    y_train = []

    # go through each image
    for root, dirs, files in os.walk(image_dir):
        if root != image_dir:
            for file in files:
                extension_set = set()
                extension_set.add(".png")
                extension_set.add(".jpg")
                extension_set.add(".jpeg")
                trash, extension = os.path.splitext((image_dir + file))
                # if it has an image extension
                if extension.lower() in extension_set:
                    path = os.path.join(root, file)
                    label = os.path.basename(root).lower()
                    # print(label, path)
                    # we have not seen this image before
                    if label not in ids:
                        ids[label] = curr_id
                        curr_id += 1

                    # even if we did see it then _id_ would be assigned to the existing value
                    _id_ = ids[label]
                    # print(ids)

                    pillow = Image.open(path).convert("L")  # gray
                    image_arr = np.array(pillow, "uint8")  # train using image array
                    faces = face_cascade.detectMultiScale(image_arr, scaleFactor=1.2, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        roi = image_arr[y:y + h, x:x + w]  # region of interest
                        x_train.append(roi)
                        y_train.append(_id_)


    # save ids into a pickle file
    with open("src/utils/ids.pickle", 'wb') as file:
        pickle.dump(ids, file)

    # train
    recognizer.train(x_train, np.array(y_train))
    recognizer.save("src/utils/trainer.yml")
