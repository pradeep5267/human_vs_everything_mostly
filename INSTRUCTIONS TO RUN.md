 ## download the trained model and pickled dataset from https://drive.google.com/open?id=1Yc6UmxVLCW544nuuwEE2AXC3LUa9xlQa download and put it in the same directory as the repo
 ## use run_inference_only.py to only run inference using the downloaded trained model.<br>
 ## use run_training_only.py to fine tune the vgg 16 model to any image dataset with binary label;the fine tuning code starts from line 113<br>
 ## download the GRAZ_01 dataset from http://www-old.emt.tugraz.at/~pinz/data/GRAZ_01/ and download the bikes_and_persons.zip then unzip it and put in the same directory as the repo<br>
 ## the dataset for persons uses the GRAZ_01 persons.zip out of which only the first 350 are used since the rest are not clear as stated in the readme of the website
 ## the non person dataset can be anything which does not include humans in the images.
 
 the run_inference function to make predictions, it takes 3 argument the trained model, path to image directory
 and the true labels(or target) of the images
 return the accuracy of the predictions and the predicted labels
 image size has to be above '224x224'
