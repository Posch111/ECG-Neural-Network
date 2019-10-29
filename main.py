#!C:\Users\bposc\OneDrive\Cardio\ECG\Convolutional Neural Network ECG (Python)\Scripts\python.exe
import os
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

# load data and chunk it into samples 1000 points long
# path='C:\\Users\\Bobby\\OneDrive\\Cardio\\ECG\\'
path = os.getcwd()
lead1f=np.loadtxt('V1\\V1_0.txt',delimiter="\n")
lead2f=np.loadtxt('V2\\V2_0.txt',delimiter="\n")
lead3f=np.loadtxt('V3\\V3_0.txt',delimiter="\n")
size=200
lead1 = [lead1f[i:i+size] for i in range(0,57000,size)]
lead2 = [lead2f[i:i+size] for i in range(0,57000,size)]
lead3 = [lead3f[i:i+size] for i in range(0,57000,size)]

# plot a sample of each lead
plt.subplot(221)
plt.plot(lead1[0])
plt.title('Lead 1')
plt.subplot(222)
plt.plot(lead2[0])
plt.title('Lead 2')
plt.subplot(223)
plt.plot(lead3[0])
plt.title('Lead 3')
plt.show()

#format and shuffle data
y1=[[0]]*len(lead1)
y2=[[1]]*len(lead2)
y3=[[2]]*len(lead3)
y=np.concatenate((y1,y2,y3))
data=np.concatenate((lead1,lead2,lead3))
data = keras.utils.normalize(data)
data = np.concatenate((data,y),axis=1)
partition=int(0.8*len(data))
# print('partition: ' + str(partition))
# print('first')
# print(data)
# # print('size of data: ' +str(len(data)))
# print('before normalize')
# print(data)

# print('after normalize')
# print(data)
data=np.random.permutation((data))
# print('after permutation')
# print(data)
# print(x)
# #separate data and create categories
Xtrain=data[0:partition,:-1]
# print('Xtrain')
# print(Xtrain)
# print('size of Xtrain: ' +str(len(Xtrain)))
Xtrain=np.reshape(Xtrain,(len(Xtrain),size,1))
# print('reshape')
# print(Xtrain)
# print(len(Xtrain))
Xpred=data[partition:,:-1]
Xpred=np.reshape(Xpred,(len(Xpred),size,1))
# print('Xpred')
# print(Xpred)
# print(len(Xpred))
Ytrain=data[0:partition,-1]
# print('Ytrain')
# print(Ytrain)
# print(len(Ytrain))
Yt=to_categorical(Ytrain)
# print('Yt')
# print(Yt)
# print(len(Yt))
Ypred=data[partition:,-1]
Yp=to_categorical(Ypred)
# print('Yp')
# print(Yp)
# print(len(Yp))

#create CNN model
model = Sequential()
model.add(Conv1D(10,10,activation='relu',input_shape=(size,1)))
model.add(Conv1D(10,10,activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(10,5,activation='relu'))
model.add(Conv1D(10,5,activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(10,2,activation='relu'))
model.add(Conv1D(10,2,activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(10,1,activation='relu'))
model.add(Conv1D(10,1,activation='relu'))
# model.add(MaxPooling1D(3))
# model.add(Dropout(0.5))
model.add((Flatten()))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(Xtrain,Yt,epochs=20)

#test model
score = model.evaluate(Xpred,Yp)
print(score)
predictions =model.predict_classes(Xpred,verbose=1)
print(predictions)
input()