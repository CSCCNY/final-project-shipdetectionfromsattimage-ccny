import xml.etree.ElementTree as XML
from itertools import repeat
from multiprocessing import Pool
from os import listdir
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


DataDir = "/media/sujoy/New Folder/Ship_Dataset/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"
AllowedNumberofImagesPerClass = {
'ship': 386,
'aircraft carrier': 101,
'warcraft': 230,
'Nimitz class aircraft carrier': 166,
'Enterprise class aircraft carrier': 180,
'Arleigh Burke class destroyers': 350,
'WhidbeyIsland class landing craft': 324 ,
'Perry class frigate': 300,
'Sanantonio class amphibious transport dock': 186,
'Ticonderoga class cruiser': 300,
'Kitty Hawk class aircraft carrier': 0,
'Admiral Kuznetsov aircraft carrier': 106,
'Abukuma-class destroyer escort': 0,
'Austen class amphibious transport dock': 281,
'Tarawa-class amphibious assault ship': 220,
'USS Blue Ridge (LCC-19)': 0,
'Container ship': 155,
'OXo|--)': 150,
'Car carrier([]==[])': 105,
'Hovercraft': 255,
'yacht': 212,
'Container ship(_|.--.--|_]=': 150,
'Cruise ship': 106,
'submarine': 200,
'lute': 125,
'Medical ship': 123,
'Car carrier(======|': 122,
'Ford-class aircraft carriers': 0,
'Midway-class aircraft carrier': 109,
'Invincible-class aircraft carrier': 0
}



imagePathPerClass = {}
ClassesNames = {}
ClassesDicts = {}
ClassesTotals = {}
TrainClassesTotals ={}
ValClassesTotals={}
RECORDS = {}
Classes = []


def getClassinfo(TrainFolder):
    global ClassesDicts,ClassesNames,ClassesTotals,Classes,TrainClassesTotals,ValClassesTotals
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
    TrainClassesTotals = dict(zip(list(ClassesNames.values()), repeat(int(0))))
    ValClassesTotals = dict(zip(list(ClassesNames.values()), repeat(int(0))))
    Classes = list(ClassesNames.keys())
    return Classes



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
    global Classes, ClassesNames,ClassesTotals,TrainClassesTotals,ValClassesTotals, imagePathPerClass
    tree = XML.parse(''.join(AnnotationsFolder + file))
    root = tree.getroot()
    annfolders = root.findall("HRSC_Objects/HRSC_Object")
    record = {}
    record["file_name"] = ''.join(PicFolder + root.findtext("Img_FileName") + '.' + root.findtext("Img_FileFmt"))
    record["image_id"] = root.findtext("Img_FileName")
    record["height"] = int(root.findtext("Img_SizeHeight"))
    record["width"] = int(root.findtext("Img_SizeWidth"))
    objs = []
    classId = None
    # print(file)
    for anno in annfolders:
        classId = anno.findtext("Class_ID")
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
            "bbox" : [x1,y1,x2,y2],
            "polygon" : poly,
            "bbox_angle": [cx, cy, w, h, a],
            "category_id": anno.findtext("Class_ID"),
            # "category_id": Classes.index(anno.findtext("Class_ID")),
        }
        if(classId != "100000004"):
            ClassesTotals[ClassesNames.get(anno.findtext("Class_ID"))] += 1
            objs.append(obj)
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

        # drawImage(record["file_name"],obj,ClassesNames.get(anno.findtext("Class_ID")))
    record["annotations"] = objs

    if classId not in imagePathPerClass:
        imagePathPerClass[classId] = [record["file_name"]]
    else: imagePathPerClass[classId].append(record["file_name"])

    return record

def showTrainValSplit():
    global ClassesTotals, TrainClassesTotals, ValClassesTotals
    plt.bar(*zip(*ClassesTotals.items()))
    plt.show()
    plt.bar(*zip(*TrainClassesTotals.items()))
    plt.show()
    plt.bar(*zip(*ValClassesTotals.items()))
    plt.show()


def returnTRAINRECORDS():
    return RECORDS["Train"]


def returnTESTRECORDS():
    return RECORDS["Test"]





def get_augmented_Ship_dicts(augmented_image_data_file,TrainData):
    import json
    global ClassesTotals
    with open(augmented_image_data_file) as json_file:
        Annotations = json.load(json_file)

        for f in Annotations:
            # annoData = fullAnnotation(f, Annotations, Images)
            annoRecord = []
            for i in range(0, len(f['annotation'])):
                fileName = f['file_name']
                height = f['height']
                width = f['width']
                startX = f['annotation'][i]['x_min']
                startY = f['annotation'][i]['y_min']
                endX = f['annotation'][i]['x_max']
                endY = f['annotation'][i]['y_max']
                classID = f['annotation'][i]['classID']
                if( ClassesTotals[ClassesNames.get(classID)] <= AllowedNumberofImagesPerClass.get(ClassesNames.get(classID)) ):
                    ClassesTotals[ClassesNames.get(classID)] += 1
                    annoRecord = [fileName,width,height,startX, startY, endX, endY, classID]
                    TrainData.append(annoRecord)
        return TrainData

def get_Ship_dicts(Dir):
    Images = ''.join(Dir + "AllImages/")
    Annotations = ''.join(Dir + "Annotations/")
    AnnotationFileList = listdir(Annotations)
    AnnoData = []
    '''
    format of annotation data: imagePath,imageID,height,width,x1,y1,x2,y2,classID
    imagePath: location of the image
    x1 = x_min
    y1 = y_min
    x2 = x_max
    y2 = y_max
    classID = corresponding class id generated from getClassInfo() 
    '''
    # annoData = fullAnnotation('100000705', Annotations, Images)
    # AnnoData.append(annoData)
    # annoRecord=[]
    # annoData = fullAnnotation('100000705.xml', Annotations, Images)
    # for i in range(0,len(annoData['annotations'])):
    #     fileName = annoData['file_name']
    #     startX = annoData['annotations'][i]['bbox'][0]
    #     startY = annoData['annotations'][i]['bbox'][1]
    #     endX = annoData['annotations'][i]['bbox'][2]
    #     endY = annoData['annotations'][i]['bbox'][3]
    #     classID = annoData['annotations'][i]['category_id']
    #     annoRecord =[fileName,startX,startY,endX,endY,classID]
    #     AnnoData.append(annoRecord)
    # print(AnnoData)

    for f in AnnotationFileList:
        annoData = fullAnnotation(f, Annotations, Images)
        annoRecord = []
        for i in range(0, len(annoData['annotations'])):
            fileName = annoData['file_name']
            height = annoData['height']
            width = annoData['width']
            startX = annoData['annotations'][i]['bbox'][0]
            startY = annoData['annotations'][i]['bbox'][1]
            endX = annoData['annotations'][i]['bbox'][2]
            endY = annoData['annotations'][i]['bbox'][3]
            classID = annoData['annotations'][i]['category_id']

            annoRecord = [fileName,width,height,startX, startY, endX, endY, classID]
            AnnoData.append(annoRecord)

    print(len(AnnoData))
    return AnnoData
    # with Pool() as p:
    #
    #     normal = list(
    #         array(p.starmap(full_annotation, zip(List, repeat(ANNOTATIONS), repeat(SEGMENTS), repeat(IMAGES)))))
    #     RECORDS[i] = list(normal)

# 5 classes has no image
# getClassinfo(TrainDir)
# AnnotationData = get_Ship_dicts(TrainDir)
# print(ClassesTotals)
# # print(ClassesNames)
# TrainData,ValidationData = get_Ship_dicts(TrainDir)
# print(TrainData[0])
# print(ValidationData[0])
# print(ClassesTotals)
# print(returnTESTRECORDS())
# dataset_dicts = returnTRAINRECORDS()
# print(ClassesTotals)