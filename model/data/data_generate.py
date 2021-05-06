
# import the necessary packages
# python generate_images.py --image dog.jpg --output generated_dataset/dogs
import model.data.dataset_mapper as dataset_mapper
import model.data.data_preprocess as data_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from dataset_mapper import fullAnnotation

DataDir = "/media/sujoy/New Folder/Ship_Dataset/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

OUTPUT = "../generatedData/"
def dataGenerator(imagePath):
	print("[INFO] loading example image...")
	image = load_img(imagePath)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
	total = 0

	# construct the actual Python generator
	print("[INFO] generating images...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir=OUTPUT,
						save_prefix="image", save_format="jpg")

	# loop over examples from our image data augmentation generator
	for image in imageGen:
		# increment our counter
		total += 1

		# if we have reached the specified number of examples, break
		# from the loop
		if total == 4:
			break

Classes = dataset_mapper.getClassinfo(TrainDir)
TrainData = dataset_mapper.get_Ship_dicts(TrainDir)
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output directory to store augmentation examples")
# ap.add_argument("-t", "--total", type=int, default=100,
# 	help="# of training samples to generate")
# args = vars(ap.parse_args())

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
# print("[INFO] loading example image...")
# image = load_img(args["image"])
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)

# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
# aug = ImageDataGenerator(
# 	rotation_range=30,
# 	zoom_range=0.15,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	shear_range=0.15,
# 	horizontal_flip=True,
# 	fill_mode="nearest")
# total = 0

# construct the actual Python generator
# print("[INFO] generating images...")
# imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
# 	save_prefix="image", save_format="jpg")
#
# # loop over examples from our image data augmentation generator
# for image in imageGen:
# 	# increment our counter
# 	total += 1
#
# 	# if we have reached the specified number of examples, break
# 	# from the loop
# 	if total == args["total"]:
# 		break