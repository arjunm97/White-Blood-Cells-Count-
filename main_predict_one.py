# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential, load_model,model_from_json
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense ,BatchNormalization
from keras import backend as Kty 
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import os
import pandas as pd 
import tkinter as tk
from tkinter import filedialog
import cv2

from sklearn.datasets import make_circles
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
img_width, img_height = 224, 224
mydir='images/test_im/im'
filetest=[f for f in os.listdir(mydir) if f.endswith(".jpg")]
for f in filetest:
    os.remove(os.path.join(mydir,f))
file_path=filedialog.askopenfilename()
im_test=cv2.imread(file_path)


train_data_dir = 'images\TRAIN'
validation_data_dir = 'images\TEST'
nb_train_samples = 9957
nb_validation_samples = 2487
epochs = 10
batch_size = 10


if Kty.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(4))
model.add(Activation(tf.nn.softmax))

model.compile(loss ='categorical_crossentropy', 
					optimizer ='rmsprop', 
				metrics =['accuracy']) 

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        horizontal_flip=True, 
        vertical_flip=False)

train_generator = datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='categorical') 

validation_generator = datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='categorical') 

checkpoint=ModelCheckpoint("model_best.hdf5",monitor='loss',verbose=1,save_best_only=True,mode='auto',period=1)

model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples // batch_size, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples // batch_size,callbacks=[checkpoint])

validation_generator = datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode =None,shuffle = False) 

#model.load_weights("model_best.hdf5")
pre=model.predict_generator(validation_generator);
preds_cls_idx = pre.argmax(axis=-1);

if(preds_cls_idx==0):
    print('EOSINOPHIL')
elif(preds_cls_idx==1):
    print('LYMPHOCYTE')
elif(preds_cls_idx==2):
    print('MONOCYTE')
elif(preds_cls_idx==2):
    print('NEUTROPHIL')
