import dataset_mapper
from data_generate import data_generator
import json

DataDir = "/media/sujoy/New Volume/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

AugmentedDir = "../generatedData/"

Classes = dataset_mapper.getClassinfo(TrainDir)
TrainData = dataset_mapper.get_Ship_dicts(TrainDir)
perClassData = dataset_mapper.ClassesTotals
classDict = dataset_mapper.ClassesNames
imagePerClass = dataset_mapper.imagePathPerClass

# print(imagePerClass)
# for imageClass in imagePerClass:
#     if(len(imagePerClass[imageClass]) < 100):
#         data_generator(imagePerClass[imageClass])
