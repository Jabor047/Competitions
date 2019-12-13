problem_dir = 'starting_kit/ingestion_program/'
score_dir = 'starting_kit/scoring_program/'
results_dir = 'results/'

import os
from sys import path; path.append(problem_dir); path.append(score_dir);
from data_io import read_as_df, write
from data_manager import DataManager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import models


data_dir = 'input_data'
data_name = 'microscopy'

data = read_as_df(data_dir + '/' + data_name)

D = DataManager(data_name, data_dir, replace_missing=True)
# print(D)

def reshape(X):
	num = X.shape[0]
	X = X.reshape((num,40,40,3))
	X = X / 255.0
	return X

X_train = reshape(D.data['X_train'])
X_test = reshape(D.data['X_test'])
X_valid = reshape(D.data['X_valid'])

Y_train = D.data['Y_train']
Y_valid = D.data['Y_valid']
Y_test = D.data['Y_test']

# if os.path.exists('saved_model/so_model_20epochsdeep.h5'):
# 	model = load_model('saved_model/so_model_20epochsdeep.h5')
# else:	
# es = EarlyStopping(monitor='val_acc',mode='max',verbose=1, patience=20)
# mc = ModelCheckpoint('saved_model/best_model.h5', monitor="val_acc",mode="max", verbose=1, save_best_only=True)
# aug = ImageDataGenerator(horizontal_flip=True,rotation_range=90,brightness_range=[0.2,1.0], shear_range=0.2, zoom_range=0.2)

# XTrain, XTest, YTrain, YTest = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# model = models.so_model()

# model.fit(X_train,Y_train, batch_size=64, shuffle=True, epochs=1000, validation_split=0.2, callbacks=[es,mc])

saved_model = load_model('saved_model/best_model.h5')
saved_model.save_weights('saved_model/best_weights.h5')
Y_hat_train = saved_model.predict(X_train) 
Y_hat_valid = saved_model.predict(X_valid)
Y_hat_test = saved_model.predict(X_test)

results_name = results_dir + data_name
write(results_name + '_train.predict', Y_hat_train)
write(results_name + '_valid.predict', Y_hat_valid)
write(results_name + '_test.predict', Y_hat_test)

metric_name, scoring_function = 'auc_binary', roc_auc_score

print('Training score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_hat_train))
# print('Valid score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_valid, Y_hat_valid))
# print('Test score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_test, Y_hat_test))
print('Ideal score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_train))

