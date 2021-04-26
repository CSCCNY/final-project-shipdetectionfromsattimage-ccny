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


def drawImage(imagePath,imageData,className):
    # Input image
    image = cv2.imread(imagePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.rectangle(image, (imageData['bbox'][0], imageData['bbox'][3]), (imageData['bbox'][1], imageData['bbox'][2]), (0, 255, 0), 2)
    box =cv2.boxPoints(
        ((imageData['bbox_angle'][0],imageData['bbox_angle'][1]),
        (imageData['bbox_angle'][2],imageData['bbox_angle'][3]),
        imageData['bbox_angle'][4]))
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 191, 255), 2)
    image = cv2.putText(image, className, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("image", image)
    cv2.waitKey()

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
        a = math.degrees(float(anno.findtext("mbox_ang")))
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
        drawImage(record["file_name"],obj,ClassesNames.get(anno.findtext("Class_ID")))
    record["annotations"] = objs
    return record


def returnTRAINRECORDS():
    return RECORDS["Train"]


def returnTESTRECORDS():
    return RECORDS["Test"]


def get_Ship_dicts(Dir):
    tree = XML.parse(Dir + 'sysdata.xml')
    root = tree.getroot()
    dataset = root.findall('HRSC_DataSet_Exp/HRSC_DSExpImgs/HRSC_DataSet_Exp_Group')
    trainImages = []
    valImages = []
    for d in dataset:
        if(d.findtext('ExpGroup_Name')=='train'):
            for k in d.findall('ExpGroup_Imgs/Img_NO'):
                fileName = k.text + 'xml'
                trainImages.append(fileName)

    print(len(trainImages))
    # Images = ''.join(Dir + "AllImages/")
    # Annotations = ''.join(Dir+"Annotations/")
    #
    # xmlFileList = listdir(Annotations)
    # data = []
    # for f in xmlFileList:
    #     imgData = fullAnnotation(f,Annotations,Images)
    #     data.append(imgData)
    # return data
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