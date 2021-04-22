
import xml.etree.ElementTree as XML
from itertools import repeat
from multiprocessing import Pool
from os import listdir
import math
import cv2
import numpy as np

DataDir = "/media/sujoy/Elements/Ship_Dataset/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

ClassesNames = {}
ClassesDicts = {}
ClassesTotals = {}
RECORDS = {}
Classes = []

def getClassinfo(TrainFolder):
    global ClassesDicts,ClassesNames,ClassesTotals,Classes
    xmlFile = 'sysdata.xml'
    ClassCount = 0
    tree = XML.parse(TrainFolder + xmlFile)
    root = tree.getroot()
    # print(LoadImageData(IMAGES))
    # Loadimages(IMAGES)
    for Shipclass in root.findall("./HRSC_Classes/HRSC_Class"):
        ClassDict = {}
        for key in Shipclass:
            ClassDict.update({key.tag: Shipclass.findtext(key.tag)})

        ClassesNames.update({Shipclass.findtext("Class_ID"): Shipclass.findtext("Class_EngName")})
        ClassesDicts.update({Shipclass.findtext("Class_ID"): ClassDict})

        ClassCount += 1
    ClassesTotals = dict(zip(list(ClassesNames.values()), repeat(int(0))))
    Classes = list(ClassesNames.keys())
    print(Classes)


def drawImage(imagePath,imageData):
    # Input image
    image = cv2.imread(imagePath)

    # Converts to grey for better reulsts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Converts to HSV
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV values
    # lower_skin = np.array([5, 36, 53])
    # upper_skin = np.array([19, 120, 125])

    # mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    # Finds contours
    # im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draws contours
    # for c in cnts:
    #     if cv2.contourArea(c) < 3000:
    #         continue

        # (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(image, (194, 507), (972, 243), (0, 255, 0), 2)
    cv2.imshow("image", image)
        ## BEGIN - draw rotated rectangle
        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(image, [box], 0, (0, 191, 255), 2)
        ## END - draw rotated rectangle

    # cv2.imwrite('out.png', image)
def fullAnnotation(file, AnnotationsFolder,PicFolder):
    global Classes, ClassesNames
    tree = XML.parse(''.join(AnnotationsFolder + file))
    root = tree.getroot()
    annfolders = root.findall("HRSC_Objects/HRSC_Object")
    record = {}
    record["file_name"] = ''.join(PicFolder + root.findtext("Img_FileName") + '.' + root.findtext("Img_FileFmt"))
    record["image_id"] = root.findtext("Img_FileName")
    record["height"] = int(root.findtext("Img_SizeHeight"))
    record["width"] = int(root.findtext("Img_SizeWidth"))
    objs = []
    # print(file)
    for anno in annfolders:
        x1 = int(anno.findtext("box_xmin"))
        x2 = int(anno.findtext("box_xmax"))
        y1 = int(anno.findtext("box_ymin"))
        y2 = int(anno.findtext("box_ymax"))
        a = -math.degrees(float(anno.findtext("mbox_ang")))
        cx = float(anno.findtext("mbox_cx"))
        cy = float(anno.findtext("mbox_cy"))
        w = float(anno.findtext("mbox_w"))
        h = float(anno.findtext("mbox_h"))
        poly = [(x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
                (x1, y1)]
        obj = {
            "bbox" : [x1,x2,y1,y2],
            "polygon" : poly,
            "bbox_angle": [cx, cy, w, h, a],
            "category_id": Classes.index(anno.findtext("Class_ID")),
        }
        ClassesTotals[ClassesNames.get(anno.findtext("Class_ID"))] += 1
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
        drawImage(record["file_name"],obj)
        break
    record["annotations"] = objs
    return record


def returnTRAINRECORDS():
    return RECORDS["Train"]


def returnTESTRECORDS():
    return RECORDS["Test"]


def get_Ship_dicts(TrainDir):

    tree = XML.parse(TrainDir + 'sysdata.xml')
    root = tree.getroot()
    Images = ''.join(TrainDir + "AllImages/")
    Annotations = ''.join(TrainDir+"Annotations/")

    xmlFileList = listdir(Annotations)
    data = []
    for f in xmlFileList:
        imgData = fullAnnotation(f,Annotations,Images)
        data.append(imgData)
    return data
    # data = list(array(map(full_annotation,zip(xmlFileList, repeat(Annotations),repeat(Images)))))
    # return list(data)
    # with Pool() as p:
    #     xmlFileList = listdir(Annotations)
    #     normal = list(
    #         np.array(p.starmap(full_annotation, zip(xmlFileList, repeat(Annotations),repeat(Images)))))
    #     return list(normal)

getClassinfo(TrainDir)
# print(ClassesNames)
TrainData = get_Ship_dicts(TrainDir)
# print(ClassesTotals)
# print(returnTESTRECORDS())
# dataset_dicts = returnTRAINRECORDS()
# print(ClassesTotals)