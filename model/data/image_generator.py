import dataset_mapper


DataDir = "/media/sujoy/New Volume/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

AugmentedDir = "../generatedData/"


perClassData = dataset_mapper.ClassesTotals
classDict = dataset_mapper.ClassesDicts
imagePerClass = dataset_mapper.imagePathPerClass
print(imagePerClass)
