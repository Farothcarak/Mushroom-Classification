#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: T.Tunç Kulaksız
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('preprocessing_mush.csv')

print(veriler.corr()["0"].sort_values())


veriler1 = veriler.iloc[:,[1,5,8,9,10,12,13,14,20,22]]

X = veriler1.iloc[:,1:10].values
Y = veriler1.iloc[:,0].values

X=np.asarray(X).astype(np.float32)
Y=np.asarray(Y).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=0)


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

model = Sequential()

model.add(Dense(64, kernel_initializer = "uniform", activation= "relu",input_dim = 9))
model.add(Dropout(0.1))

model.add(Dense(16, kernel_initializer = "uniform", activation= "relu"))
model.add(Dropout(0.3))

model.add(Dense(32, kernel_initializer = "uniform", activation= "relu"))
model.add(Dropout(0.2))

model.add(Dense(64, kernel_initializer = "uniform", activation= "relu"))
model.add(Dropout(0.1))

model.add(Dense(16, kernel_initializer = "uniform", activation= "relu"))
model.add(Dropout(0.3))

model.add(Dense(1, activation= "sigmoid"))

model.compile(optimizer="adam",loss = "binary_crossentropy",metrics = ["accuracy"])

earlystopping = EarlyStopping (monitor="val_loss",mode="min",verbose =1, patience = 25)

model.fit(X_train,y_train,epochs = 100, batch_size = 1024, validation_data=(X_test,y_test),verbose=1,callbacks=[earlystopping])

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)

"""
from tensorflow.keras.models import load_model

model.save("pred_mushroom.h5")
"""


