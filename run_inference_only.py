
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pandas as pd

import os
import itertools
import pickle

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


from keras.utils import to_categorical,normalize

from sklearn.model_selection import train_test_split

from keras.models import load_model

from keras.applications import vgg16

from keras import layers, models

import random

from sklearn.metrics import accuracy_score

#USE THIS CODE BLOCK TO LOAD IN PREPROCESSED TRAINING DATA

pickle_in = open("mixed_human_classification_y.pickle","rb")
Y = pickle.load(pickle_in)

pickle_in = open("mixed_human_classification_X_img.pickle","rb")
X_img = pickle.load(pickle_in)

#Open the pickle file
#Use pickle.load() to load it to a var.#USE THIS BLOCK TO LOAD IN PRETRAINED MODEL
model = load_model('human_classifier_vgg16.h5')
#%%
# USE run_inference function to make predictions, it takes 3 argument the trained model, path to image directory
# and the true labels(or target) of the images
# return the accuracy of the predictions and the predicted labels
# image size has to be above '224x224'
def run_inference(model,path,label):
    IMG_SIZE = 224
    predicted_data = {}
    count = 0
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path,img))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))            
            new_array = new_array/255.0
            training_data.append([new_array,label])
            #y.append(1)
            count = count + 1
        except Exception as e:  # in the interest in keeping the output clean...
            pass
    input_img = []
    input_target = []

    for features,label in training_data:
        input_img.append(features)
        input_target.append(label)
    input_img_arr = np.array(input_img).reshape(-1,224,224,3)

    predictions = model.predict_classes(input_img_arr)
    x = accuracy_score(input_target, predictions)
    predicted_data = {'accuracy' : x,
                      'predicted_labels' : predictions}
    return(predicted_data)
#%%
Y = run_inference(model,path,label)