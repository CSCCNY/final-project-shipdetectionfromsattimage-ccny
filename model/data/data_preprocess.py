import dataset_mapper

DataDir = "/media/sujoy/Elements/Ship_Dataset/HRSC2016/"
TrainDir = DataDir + "Train/"
TestDir = DataDir + "Test/"

Classes = dataset_mapper.getClassinfo(TrainDir)
TrainData = dataset_mapper.get_Ship_dicts(TrainDir)
print(len(TrainData))