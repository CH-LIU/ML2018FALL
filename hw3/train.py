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
##################################### read file #####################################
train_x_file = sys.argv[1]
table_x = pd.read_csv(train_x_file , encoding = 'big5')
table_x = pd.DataFrame(table_x)


############################### data processing ############################################

x_feature = np.array(table_x)
tempp = []
for i in x_feature:
    temp = i[1].split(' ')
    tempp.append(temp)
x_feature =  np.array(tempp).astype(float) 
label_onehot = pd.get_dummies(table_x['label'])
label_onehot = np.array(label_onehot).astype(float)

ran = np.hstack((label_onehot , x_feature))
np.random.shuffle(ran)

label_onehot = ran[ : , 0:7]
x_feature = ran[ : , 7 : ]

x_feature =  x_feature.reshape(28709 , 48 , 48 ,1)
x_feature = x_feature / 255


x_train = x_feature[5742 : , : , :]
x_validate = x_feature[0 : 5742 , : , :]


y_train = label_onehot[5742 : , :]
y_validate = label_onehot[0 : 5742 , :]


########################## image augmentation ######################################################

datagen = ImageDataGenerator(rotation_range = 25,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              shear_range = 0.2,
                              zoom_range = [0.8, 1.2],
                              horizontal_flip = True)
datagen.fit(x_train)




##########################  building model #################################################
model = Sequential()
model.add(Conv2D(64,(3,3), padding='same', input_shape = (48,48,1)))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D((2,2), padding='same'))
model.add(Dropout(0.2))
'''model.add(Conv2D(128,(3,3) , padding='same'))
model.add(Activation('relu'))'''
model.add(Conv2D(128,(3,3) , padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D((2,2) , padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(256,(3,3) , padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D((2,2) , padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(512,(3,3) , padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D((2,2) , padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(output_dim = 256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(output_dim = 7))
model.add(Activation('softmax'))

model.summary()

########################## training ################################################
#Adam(lr=0.0005 , decay = 0.0000001 )
#SGD(lr=0.005, decay=0.00001, momentum=0.9)
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
checkpointer = ModelCheckpoint(filepath="phi.h5", monitor='val_acc' , verbose=1, save_best_only=True)
#training = model.fit(x = x_train , y = y_train  , validation_data = (x_validate , y_validate) , epochs = 100 , batch_size = 64 , verbose = 1)
training = model.fit_generator(datagen.flow(x_train , y_train , batch_size = 128) , steps_per_epoch = 10*x_train.shape[0]//128 ,  epochs = 500 , callbacks=[checkpointer] , verbose = 1 ,
validation_data = (x_validate , y_validate))

########################## testing ################################################
model = load_model("phi.h5")


test_x_file = sys.argv[2]
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