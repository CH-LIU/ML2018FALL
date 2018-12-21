import sys
import pandas as pd
import numpy as np
import jieba
import gensim
import h5py
from keras.models import Sequential, Model  , load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Input , BatchNormalization
from keras.optimizers import Adam , SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

jieba.load_userdict(sys.argv[3])
my_word2vec_model = gensim.models.Word2Vec.load('word2vec_256.model')
##################################### read file #####################################
train_x_file = sys.argv[1]
table_x = pd.read_csv(train_x_file  , sep = 'delimiter' , encoding = 'utf-8' )
table_x = pd.DataFrame(table_x)
table_x = np.array(table_x)

table_y = pd.read_csv(sys.argv[2] , encoding = 'utf-8' )
table_y = pd.DataFrame(table_y)
y_train = np.array(table_y).astype(float)
y_train = y_train[: , 1:]

############################### data processing ############################################
'''test_x_file = sys.argv[3]
table_x2 = pd.read_csv(test_x_file  , sep = 'delimiter' , encoding = 'utf-8' )
table_x2 = pd.DataFrame(table_x2)
table_x2 = np.array(table_x2)

word2vec_data = np.vstack((table_x , table_x2))
sentence_list = []
for comment in word2vec_data:
	s = comment[0].split(',' , 1)[1]
	s_cut = jieba.cut(s)
	temp = []
	for i in s_cut : 
		if i != ''  and i != ' ': temp.append(i)
	sentence_list.append(temp)
for i in range(7):	
    sentence_list.append(["oov"])
    sentence_list.append([" "])

word2vec_model = gensim.models.word2vec.Word2Vec(sentence_list, size = 256 , min_count = 7 , iter = 10 )
word2vec_model.save('my_word2vec_model_256_false.model')'''
x_train = []
for comment in table_x:
	s = comment[0].split(',' , 1)[1]
	s_cut = jieba.cut(s)
	temp = []
	for word in s_cut :
		if  word in my_word2vec_model.wv.vocab : 
			temp.append(my_word2vec_model[word])
		#else : temp.append(my_word2vec_model["oov"])

	x_train.append(temp)

x_train = pad_sequences(x_train, maxlen = 48 , dtype='int32', padding='post', truncating='post', value = my_word2vec_model[" "])
##########################  building model #################################################
model =  Sequential()
model.add(LSTM(256 , return_sequences = True , input_length = 48 , input_dim = 256 , dropout=0.5 , recurrent_dropout=0.5 , kernel_initializer='he_normal'))
model.add(LSTM(256 , return_sequences = False , input_length = 48 , input_dim = 256 , dropout=0.5 , recurrent_dropout=0.5 , kernel_initializer='he_normal'))
#model.add(Flatten())
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(output_dim = 512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(output_dim = 1))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss = 'binary_crossentropy' , optimizer = "adam" , metrics = ['accuracy'])
checkpointer = ModelCheckpoint(filepath = "karen.h5", monitor = 'val_acc' , verbose = 1, save_best_only = True)
training = model.fit(x = x_train , y = y_train  , validation_split = 0.2 , callbacks=[checkpointer] , epochs = 100 , batch_size = 64 , verbose = 1 , shuffle = True)

