
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import numpy as np


# DataDir = "/media/sujoy/Elements/Ship_Dataset/HRSC2016/"
# TrainDir = DataDir + "Train/"
# TestDir = DataDir + "Test/"

imageData = []
classLabels = []
bboxes = []
imagePaths = []

def preprocess(TrainData):
    global imageData,classLabels,bboxes,imagePaths
    for i in TrainData:
        # print(i)
        # normalize x1,y1,x2,y2 with respect to height and width
        startX = float(i[3]) / float(i[1])
        startY = float(i[4]) / float(i[2])
        endX = float(i[5]) / float(i[1])
        endY = float(i[6]) / float(i[2])

        # load the image and preprocess it
        image = load_img(i[0], target_size=(224, 224))
        image = img_to_array(image)

        # update our list of data, class labels, bounding boxes, and
        # image paths
        imageData.append(image)
        classLabels.append(i[7])
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(i[0])

    imageData = np.array(imageData, dtype="float32") / 255.0
    classLabels = np.array(classLabels)
    bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)
    classLabels = to_categorical(classLabels)
    return imageData,classLabels,bboxes,imagePaths,classLabels


# preprocess(TrainData)
# print(len(TrainData),len(imageData),len(classLabels),len(bboxes),len(imagePaths))