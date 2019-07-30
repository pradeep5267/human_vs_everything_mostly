
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
#%%
label_list = []
i = 0
training_data = []
Y = []
y = []
X = []
x = []
#%%
#CHANGE THE DIRECTORY PATH ACCORDING TO YOUR MACHINE
path_human = './persons'
path_non_human = './non_persons'
IMG_SIZE = 224


#%%
count = 0
for img in tqdm(os.listdir(path_human)):
    if count <= 350:
        try:
            #img_array = cv2.imread(os.path.join(path_human,img),cv2.IMREAD_GRAYSCALE)
            img_array = cv2.imread(os.path.join(path_human,img))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))            
            new_array = new_array/255.0
            training_data.append([new_array,1])
            #y.append(1)
            count = count + 1
        except Exception as e:  # in the interest in keeping the output clean...
            pass

count = 0
for img in tqdm(os.listdir(path_non_human)):
    if count >= 0:
        try:
            img_array = cv2.imread(os.path.join(path_non_human,img)) #,cv2.IMREAD_GRAYSCALE
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))            
            new_array = new_array/255.0
            training_data.append([new_array,0])
            #y.append(0)
            count = count + 1
        except Exception as e:  # in the interest in keeping the output clean...
            pass


#%%
random.shuffle(training_data)
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)


ip = X
pickle_out = open("input.pickle","wb")
pickle.dump(ip, pickle_out)
pickle_out.close()
ip = y
pickle_out = open("output.pickle","wb")
pickle.dump(ip, pickle_out)
pickle_out.close()
#for debugging/sanity check
# print(len(training_data[1]))
# print(len(X))
# print(len(y))

# print(y[35])
# print(y[359])

# print(Y[35])
# print(Y[359])
#%%
X_img = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
#%%
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_img, y, test_size=0.1, random_state=42)

#for debugging
# print(X_img.shape)

# print(y.shape)
#%%
# fine tuning starts here
###############################################################################
vgg16_model = vgg16.VGG16()
type(vgg16_model)


#%%
funcmodel = Sequential()
#for layer in vgg16_model.layers:
#    print(layer)


#%%
input_shape = (350, 256, 256, 3)
input_layer = layers.Input(batch_shape=vgg16_model.layers[0].input_shape)
prev_layer = input_layer
for layer in vgg16_model.layers:
    prev_layer = layer(prev_layer)
funcmodel = models.Model([input_layer], [prev_layer])

#FOR DEBUGGING
#funcmodel.summary()
#%%
#REMOVE THE LAST LAYER IE THE PREDCTION LAYER OF VGG16 WHICH HAS multi CLASS PREDICTION LAYER
funcmodel.layers.pop()

#FOR DEBUGGING
#funcmodel.summary()
#%%
for layer in funcmodel.layers:
    layer.trainable = False


#%%
# Create the model
model1 = models.Sequential()
 
# Add the vgg base model without the final layer
model1.add(vgg16_model)


#%%
# Add new layers

model1.add(layers.Dense(1024, activation='relu'))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(2, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model1.summary()


#%%
model1.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


#%%
model1.fit(X_train_cnn, y_train_cnn, batch_size=25, epochs=15, verbose = 1)


#%%
model1.save('human_classifier_vgg16.h5')


#%%
predictions = model1.predict_classes(X_test_cnn)


#%%
x = accuracy_score(y_test_cnn, predictions)
# x