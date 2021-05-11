from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import os
import imutils
import pickle
import cv2
import os
MODEL_PATH = "/media/sujoy/New Volume/Github/final-project-shipdetectionfromsattimage-ccny/model/model_weights/"
DataDir = "/media/sujoy/New Volume/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/AllImages/"

print("[INFO] loading object detector...")
imagePaths = os.listdir(TestDir)
model = load_model(MODEL_PATH+"balanced_aug_basic_model_b2_e50.h5")
lb = pickle.loads(open(MODEL_PATH+"balanced_aug_basic_model_b2_e50.pickle", "rb").read())

for imagePath in imagePaths:
    # load the input image (in Keras format) from disk and preprocess
    # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(TestDir+imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # predict the bounding box of the object along with the class
    # label
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    # determine the class label with the largest predicted
    # probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    image = cv2.imread(TestDir+imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY),
              (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)