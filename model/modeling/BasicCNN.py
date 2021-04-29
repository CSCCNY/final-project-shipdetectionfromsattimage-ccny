'''
-------------------------------
Network Architecture
1.Convolution, Filter shape:(5,5,6), Stride=1, Padding=’SAME’
2. Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’
3. ReLU
4. Convolution, Filter shape:(5,5,16), Stride=1, Padding=’SAME’
5. Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’
6. ReLU
7. Fully Connected Layer (128)
8. ReLU
9. Fully Connected Layer (10)
10. Softmax
------------------------------------------
'''
import os
from tensorflow.keras.optimizers import Adam
import model.data.dataset_mapper as dataset_mapper
import model.data.data_preprocess as data_preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle

DataDir = "/media/sujoy/New Volume1/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

Classes = dataset_mapper.getClassinfo(TrainDir)
TrainData = dataset_mapper.get_Ship_dicts(TrainDir)

data,labels,bboxes,imagePaths,lb= data_preprocess.preprocess(TrainData)


split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.20, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# # write the testing image paths to disk so that we can use then
# # when evaluating/testing our object detector
# print("[INFO] saving testing image paths...")
# f = open(config.TEST_PATHS, "w")
# f.write("\n".join(testPaths))
# f.close()
#
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
#
# # freeze all VGG layers so they will *not* be updated during the
# # training process
vgg.trainable = False
#
# # flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
#
# # construct a fully-connected layer header to output the predicted
# # bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid",name="bounding_box")(bboxHead)
#
# # construct a second fully-connected layer head, this one to predict
# # the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)
#
# # put together our model which accept an input image and then output
# # bounding box coordinates and a class label
model = Model(
	inputs=vgg.input,
	outputs=(bboxHead, softmaxHead))
#
# # define a dictionary to set the loss methods -- categorical
# # cross-entropy for the class label head and mean absolute error
# # for the bounding box head
losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error",
}
#
# # define a dictionary that specifies the weights per loss (both the
# # class label and bounding box outputs will receive equal weight)
lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}
#
# # initialize the optimizer, compile the model, and show the model
# # summary
opt = Adam(lr=0.0001)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
# print(model.summary())
Batch_Size=32
Num_Epoch=100
# checkpoint_path = "../checkpoint/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# cp_callback = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True, verbose=1, save_freq=1*Batch_Size)

# model.save_weights(checkpoint_path.format(epoch=Num_Epoch))

# construct a dictionary for our target training outputs
trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}

# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}

# train the network for bounding box regression and class label
# prediction
print("training model...")
# H = load_model('../model_weights/loaded_model.h5')
# H.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=Batch_Size,
	epochs=Num_Epoch,
	verbose=1)

# serialize the model to disk
print("saving object detector model...")
model.save('../model_weights/basic_model100.h5', save_format="h5")

# serialize the label binarizer to disk
print("saving label...")
f = open('../model_weights/basic_model100.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, Num_Epoch)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join(["../plots/", "losses100.png"])
plt.savefig(plotPath)
plt.close()

# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"], label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# save the accuracies plot
plotPath = os.path.sep.join(["../plots/", "accs100.png"])
plt.savefig(plotPath)
