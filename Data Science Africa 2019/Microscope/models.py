
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model,Sequential
from keras.layers import Flatten,Dense,Dropout,Activation,Input,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers


def so_model():
	model = Sequential()

	model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', input_shape=(40,40,3),activation="relu",kernel_regularizer=regularizers.l2(0.008)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
	model.add(Dropout(0.1))

	model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid',activation="relu",kernel_regularizer=regularizers.l2(0.008)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
	model.add(Dropout(0.1))

	model.add(Conv2D(96, kernel_size=(3,3), strides=(1,1), padding='valid',activation="relu",kernel_regularizer=regularizers.l2(0.008)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1), padding='valid'))
	model.add(Dropout(0.1))

	model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='valid',activation="relu",kernel_regularizer=regularizers.l2(0.008)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='valid'))
	model.add(Dropout(0.1))

	model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
	model.add(Dropout(0.1))

	# model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', kernel_regularizer=regularizers.l2(0.008)))
	# model.add(BatchNormalization())
	# model.add(Activation('relu'))
	# model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
	# model.add(Dropout(0.1))

	# model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='valid'))
	# model.add(BatchNormalization())
	# model.add(Activation('relu'))
	# model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='valid' ))
	# model.add(Dropout(0.1))

	# model.add(Conv2D(1024, kernel_size=(3,3), strides=(1,1), padding='valid'))
	# model.add(BatchNormalization())
	# model.add(Activation('relu'))
	# model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='valid' ))
	# model.add(Dropout(0.1))
	#,kernel_regularizer=regularizers.l2(0.008)

	model.add(Flatten())
	model.add(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.008)))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(1, activation='relu'))
	

	model.add(Activation('sigmoid'))

	optim = Adam(lr= 1e-03 , decay=1e-03/50)
	model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

	model.summary()

	return model