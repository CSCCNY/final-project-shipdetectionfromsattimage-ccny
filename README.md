# final-project-shipdetectionfromsattimage-ccny


## Project Structure
* eda : Exploratory data analysis
* meeting_minutes : Meeting for the project.
* model: Code for the project
  * data : data preprocess code
  * evaluation : evaluation code
  * modelweights : modelweights
  * modeling: BasicCNN and FastRCNN codes
  * plots: Plots on the model
* research notes: research notes on the project.

## model/data folder
* image_generator.py generates augmeneted images using data_generate.py
* dataset_mapper.py extracts bounding box and image information from the annotations and image files.

## model/modeling folder
* BasicCNN.py and BasicCNNPredictor uses dataset_mapper to get the training data. then uses CNN model to train the data and predict the data.
* FastRCNN and FastRCNNtest uses dataset_mapper to get the training data. then uses Fast R CNN model to train the data.

## Run the code
'''
pip install -r requrements.txt
python model/modeling/BasicCNN.py

'''
