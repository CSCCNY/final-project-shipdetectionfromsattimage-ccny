
# import the necessary packages
# python generate_images.py --image dog.jpg --output generated_dataset/dogs
import model.data.dataset_mapper as dataset_mapper
import model.data.data_preprocess as data_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from dataset_mapper import fullAnnotation
import xml.etree.ElementTree as XML
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import random

DataDir = "/media/sujoy/New Volume/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

OUTPUT = "../generatedData/"

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
	img = image.copy()
	for bbox, category_id in zip(bboxes, category_ids):
		class_name = category_id_to_name[category_id]
		img = visualize_bbox(img, bbox, class_name)
		plt.figure(figsize=(12, 12))
		plt.axis('off')
		plt.imshow(img)
		plt.show()



def getAnnotation(file):

	tree = XML.parse(file)
	root = tree.getroot()
	annfolders = root.findall("HRSC_Objects/HRSC_Object")
	record = {}
	record["image_id"] = root.findtext("Img_FileName")
	record["height"] = int(root.findtext("Img_SizeHeight"))
	record["width"] = int(root.findtext("Img_SizeWidth"))
	objs = []
	classId = None
	max_w=0
	max_h=0
	# print(file)
	for anno in annfolders:
		x1 = int(anno.findtext("box_xmin"))
		x2 = int(anno.findtext("box_xmax"))
		y1 = int(anno.findtext("box_ymin"))
		y2 = int(anno.findtext("box_ymax"))
		# a = math.degrees(float(anno.findtext("mbox_ang")))
		# cx = float(anno.findtext("mbox_cx"))
		# cy = float(anno.findtext("mbox_cy"))
		# w = float(anno.findtext("mbox_w"))
		# h = float(anno.findtext("mbox_h"))
		# max_w = w if w > max_w else max_w
		# max_h = h if h > max_h else max_h
		obj = {
			"bbox" : [x1,y1,(x2-x1),(y2-y1)],
			# "polygon" : poly,
			# "bbox_angle": [cx, cy, w, h, a],
			"category_id": anno.findtext("Class_ID"),
			# "category_id": Classes.index(anno.findtext("Class_ID")),
		}


        # if(isTrain): TrainClassesTotals[ClassesNames.get(anno.findtext("Class_ID"))] += 1
        # else: ValClassesTotals[ClassesNames.get(anno.findtext("Class_ID"))] += 1
        # for key in ClassesNames.keys():
        ##  obj.update({key:0})
        # obj.update({obj["category_id"]:1})
        # try:
        #     pass
        # # obj["bbox"]=np.array(obj["bbox"],dtype=np.int64)
        # except:
        #     print(file + ":")
        #     print(obj["bbox"])
		objs.append(obj)
        # drawImage(record["file_name"],obj,ClassesNames.get(anno.findtext("Class_ID")))
	record["annotations"] = objs
	# record["max_w"] = max_w
	# record["max_h"] = max_h
	return record



def getBoundingBox(imagePath):
	annoFile = TrainDir + "Annotations/" + str.split((str.split(imagePath,'/')[-1]),'.')[0] + '.xml'
	r = getAnnotation(annoFile)
	bboxes=[]
	category_ids = []
	bbox_sizes=[]
	for anno in r['annotations']:
		bboxes.append(anno['bbox'])
		category_ids.append(anno['category_id'])
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	visualize(image, bboxes, category_ids, dataset_mapper.ClassesNames)
	return bboxes,category_ids,r['width'],r['height']



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
perClassData = dataset_mapper.ClassesTotals
classDict = dataset_mapper.ClassesDicts
classesNames = dataset_mapper.ClassesNames
imagePerClass = dataset_mapper.imagePathPerClass
print(perClassData)

for i in imagePerClass:
	for k in range(0, len(imagePerClass[i])):
		name= '/media/sujoy/New Volume/HRSC2016/Train/AllImages/100000637.bmp'
		# bboxes,category_ids,max_w,max_h = getBoundingBox(imagePerClass[i][k])
		bboxes, category_ids, max_w, max_h = getBoundingBox(name)

		# transform = A.Compose([
		# 	A.RandomSizedBBoxSafeCrop(width=700, height=700, erosion_rate=0.2,
		# 	A.HorizontalFlip(p=0.7),
		# 	A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.8),
		# ])
		transform = A.Compose( [
				A.RandomSizedBBoxSafeCrop(width=int(max_w),height=int(max_h),erosion_rate=0.2),
				A.RandomScale(p=0.8),
				A.Flip(p=0.7),
				A.RandomBrightnessContrast(brightness_limit=.7, contrast_limit=.7, p=0.8)
			],

			bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
		)
		# counter=7
		# while(counter < 50):

		image = cv2.imread(name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

		# another_transformed_image = transform(image=image, bboxes=bboxes, category_ids=category_ids)
		# another_transformed_image1 = transform(image=image, bboxes=bboxes, category_ids=category_ids)
		#
		visualize(
			transformed['image'],
			transformed['bboxes'],
			transformed['category_ids'],
			dataset_mapper.ClassesNames
		)
		cv2.imwrite("abcd.bmp", transformed['image'])


		# visualize(
		# 	another_transformed_image['image'],
		# 	another_transformed_image['bboxes'],
		# 	another_transformed_image['category_ids'],
		# 	dataset_mapper.ClassesNames
		# )
		# visualize(
		# 	another_transformed_image1['image'],
		# 	another_transformed_image1['bboxes'],
		# 	another_transformed_image1['category_ids'],
		# 	dataset_mapper.ClassesNames
		# )
		# 	# print(counter)
		# 	# counter +=7

		break
	break



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
# 	if total == args[