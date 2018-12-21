import sys
import pandas as pd
import numpy as np
import jieba
import gensim
import h5py
from keras.models import Sequential, Model , load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Input , BatchNormalization
from keras.optimizers import Adam , SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

jieba.load_userdict(sys.argv[3])
my_word2vec_model = gensim.models.Word2Vec.load('word2vec_256.model')
model = load_model(sys.argv[2])
##################################### read file #####################################
test_x_file = sys.argv[1]
table_x = pd.read_csv(test_x_file  , sep = 'delimiter' , encoding = 'utf-8' )
table_x = pd.DataFrame(table_x)
table_x = np.array(table_x)

############################### data processing ############################################
sentence_list = []
for comment in table_x:
	s = comment[0].split(',' , 1)[1]
	s_cut = jieba.cut(s)
	temp = []
	for i in s_cut : 
		if i != '' : 
			if i in my_word2vec_model.wv.vocab : 
				temp.append(my_word2vec_model[i])
		#	else : temp.append(my_word2vec_model["oov"])	
	sentence_list.append(temp)

x_test = pad_sequences(sentence_list , maxlen = 48, dtype='int32', padding='post', truncating='post', value = my_word2vec_model[" "])



y_predict = model.predict(x_test)
#y = y_predict.argmax(axis = -1).reshape(80000,1)
y = []
for i in y_predict:
	if i[0] >= 0.5 : 
		y.append(1)
	else : y.append(0)
y = np.array(y).reshape(80000,1)

index = np.array([[str(i)] for i in range(80000)])
solution = np.hstack((index,y))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'label']
solution.to_csv(sys.argv[4] , columns = ['id' , 'label'] ,  index = False , sep = ',')

