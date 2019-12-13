problem_dir = 'starting_kit/ingestion_program/'
score_dir = 'starting_kit/scoring_program/'
results_dir = 'results/'
model_count_dir = 'starting_kit/sample_code_submission/'

from sys import path; path.append(problem_dir); path.append(score_dir); path.append(model_count_dir);
from data_io import read_as_df, write
from data_manager import DataManager
from model_count import model_count
from sklearn.metrics import r2_score
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import models

data_dir = 'input_data'
data_name= 'microscopyCount'

D = DataManager(data_name, data_dir, replace_missing=True)

X_train = D.data['X_train']
Y_train = D.data['Y_train']
X_valid = D.data['X_valid']
Y_valid = D.data['Y_valid']
X_test = D.data['X_test']
Y_test = D.data['Y_test']

so_model = load_model('saved_model/best_model.h5')

so_model_pred = model_count(so_model)

so_model_pred.fit(X_train,Y_train)


Y_hat_train = so_model_pred.predict(X_train) 
Y_hat_valid = so_model_pred.predict(X_valid)
Y_hat_test = so_model_pred.predict(X_test)

results_name = results_dir + data_name
write(results_name + '_train.predict', Y_hat_train)
write(results_name + '_valid.predict', Y_hat_valid)
write(results_name + '_test.predict', Y_hat_test)

metric_name, scoring_function = 'r2_regression', r2_score

print('Training score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_hat_train))
# print('Valid score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_valid, Y_hat_valid))
# print('Test score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_test, Y_hat_test))
print('Ideal score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_train, Y_train))
