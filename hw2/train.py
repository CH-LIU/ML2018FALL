import pandas as pd
import numpy as np
import sys
import math
import time 

start = time.time()
##################################### read file #####################################
train_x_file = sys.argv[1]
table_x = pd.read_csv(train_x_file , encoding = 'big5')
table_x = pd.DataFrame(table_x)

train_y_file = sys.argv[2]
table_y = pd.read_csv(train_y_file , encoding = 'big5')
table_y = pd.DataFrame(table_y)


############################### data processing ############################################

x_train = np.array(table_x)
x_train = x_train.astype(np.float)
y_train = np.array(table_y)
y = y_train.astype(np.float)




x1 = []
x0 = []
for i in range(20000):
    if y[i] == 1 : x1.append(x_train[i])
    else : 	x0.append(x_train[i])

mean1 = np.mean(x1 , axis = 0)
mean0 = np.mean(x0 , axis = 0)
x1 = np.array(x1) #4445
x0 = np.array(x0) #15555 

cov1 = np.zeros(529).reshape(23,23)
for i in x1:
	cov1 += (i - mean1).reshape(23,1).dot((i - mean1).reshape(1,23))
cov1 /= 4445

cov0 = np.zeros(529).reshape(23,23)

for i in x1:
	cov0 += (i - mean0).reshape(23,1).dot((i - mean0).reshape(1,23))
cov0 /= 15555

cov = cov1*4445/20000 + cov0*15555/20000

test_x_file = sys.argv[3]
table_x = pd.read_csv(test_x_file , encoding = 'big5')
table_x = pd.DataFrame(table_x)

x_test = np.array(table_x)
x_test = x_test.astype(np.float)

inv_cov = np.linalg.inv(cov)

ans = []
for i in range(10000):
	 z = (mean1-mean0).T.dot(inv_cov).dot(x_test[i]) - 0.5*mean1.T.dot(inv_cov).dot(mean1) + 0.5*mean0.T.dot(inv_cov).dot(mean0) + np.log(4445/15555)
	 prob = 1 / (1 + np.exp(-z))
	 if prob >= 0.5 : ans.append(1)
	 else : ans.append(0)

ans = np.array(ans).reshape(10000 , 1)
index = np.array([["id_" + str(i)] for i in range(10000)])
solution = np.hstack((index,ans))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'Value']
solution.to_csv(sys.argv[4] , columns = ['id' , 'Value'] ,  index = False , sep = ',')	 
