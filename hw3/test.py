import pandas as pd
import numpy as np
import sys
import math 
from keras.models import Sequential , load_model
from keras.layers import Dense, Dropout, Activation, Flatten , LeakyReLU , BatchNormalization
from keras.layers.convolutional import Conv2D 
from keras.layers.pooling import MaxPooling2D , AveragePooling2D
from keras.optimizers import Adam , SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import h5py




model = load_model(sys.argv[2])


test_x_file = sys.argv[1]
test_x = pd.read_csv(test_x_file , encoding = 'big5')
test_x = pd.DataFrame(test_x)
test_x = np.array(test_x)
temppp = []
for i in test_x:
    temp = i[1].split(' ')
    temppp.append(temp)
x_test = np.array(temppp).reshape(7178 , 48 , 48 ,1).astype(float)
x_test = x_test/255

y_predict = model.predict(x_test)
y = y_predict.argmax(axis = -1).reshape(7178,1)




index = np.array([[str(i)] for i in range(7178)])
solution = np.hstack((index,y))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'label']
solution.to_csv(sys.argv[3] , columns = ['id' , 'label'] ,  index = False , sep = ',')