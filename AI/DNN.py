import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

over_sampler = RandomOverSampler()
ada = ADASYN()
SMOTE = SMOTE()

X_train = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/X_train.csv')
Y_train = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/Y_train.csv')
X_test = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/X_test.csv')
Y_test = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/Y_test.csv')

X_train = X_train.iloc[:, 1:]
Y_train = Y_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]
Y_test = Y_test.iloc[:, 1:]

X_train_RandOS, Y_train_RandOS = over_sampler.fit_resample(X_train, Y_train)
X_train_ada, Y_train_ada = ada.fit_resample(X_train, Y_train)
X_train_smote, Y_train_smote = SMOTE.fit_resample(X_train, Y_train)

X_test = X_test.values
Y_test = Y_test.values
X_train_smote = X_train_smote.values
Y_train_smote = Y_train_smote.values
X_train_ada = X_train_ada.values
Y_train_ada = Y_train_ada.values
X_train_RandOS = X_train_RandOS.values
Y_train_RandOS = Y_train_RandOS.values

model = keras.Sequential()
model.add(keras.layers.Dense(45, activation = 'relu', input_shape = (30,)))
model.add(keras.layers.Dense(45,activation = 'relu'))
model.add(keras.layers.Dense(45,activation = 'relu'))
model.add(keras.layers.Dense(45,activation = 'relu'))
model.add(keras.layers.Dense(2, activation = 'sigmoid'))
# 'rmsprop'는 optimizer default값
model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.summary()

over_sampler = RandomOverSampler()
ada = ADASYN()
SMOTE = SMOTE()

X_train = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/X_train.csv')
Y_train = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/Y_train.csv')
X_test = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/X_test.csv')
Y_test = pd.read_csv('/Users/harock96/Documents/2021년 1학기/인공지능/프로젝트/데이터 전처리/Y_test.csv')

X_train = X_train.iloc[:, 1:]
Y_train = Y_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]
Y_test = Y_test.iloc[:, 1:]

X_train_RandOS, Y_train_RandOS = over_sampler.fit_resample(X_train, Y_train)
X_train_ada, Y_train_ada = ada.fit_resample(X_train, Y_train)
X_train_smote, Y_train_smote = SMOTE.fit_resample(X_train, Y_train)

X_test = X_test.values
Y_test = Y_test.values
X_train_smote = X_train_smote.values
Y_train_smote = Y_train_smote.values
X_train_ada = X_train_ada.values
Y_train_ada = Y_train_ada.values
X_train_RandOS = X_train_RandOS.values
Y_train_RandOS = Y_train_RandOS.values

model = keras.Sequential()
model.add(keras.layers.Dense(45, activation = 'relu', input_shape = (30,)))
model.add(keras.layers.Dense(45,activation = 'relu'))
model.add(keras.layers.Dense(45,activation = 'relu'))
model.add(keras.layers.Dense(45,activation = 'relu'))
model.add(keras.layers.Dense(2, activation = 'sigmoid'))
# 'rmsprop'는 optimizer default값
model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
model.summary()

model.fit(X_train_smote, Y_train_smote, epochs = 10)

model.evaluate(X_test, Y_test)

history = model.fit(X_train_smote, Y_train_smote, epochs = 10 , verbose = 0, validation_data = (X_test, Y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()