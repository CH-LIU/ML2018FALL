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
table_x = table_x.drop(['SEX' , 'AGE'] , axis = 1)


x_test = np.array(table_x)
x_test = x_test.astype(np.float)
bias = np.ones(10000).reshape(10000,1).astype(np.float)

# EDUCATION(1) 6
one_hot_e = pd.get_dummies(table_x['EDUCATION'])
one_hot_e = np.array(one_hot_e)
one_hot_e = one_hot_e[ : , 1:]

#MARRIAGE(2) 3
one_hot_m = pd.get_dummies(table_x['MARRIAGE'])
one_hot_m = np.array(one_hot_m)
one_hot_m = one_hot_m[ : , 1:]

#PAY0(3) 11
one_hot_pay0 = pd.get_dummies(table_x['PAY_0'])
one_hot_pay0 = np.array(one_hot_pay0)

#PAY2(4) 11
one_hot_pay2 = pd.get_dummies(table_x['PAY_2'])
one_hot_pay2 = np.array(one_hot_pay2)

#PAY3(5) 10 + 1(8)
one_hot_pay3 = pd.get_dummies(table_x['PAY_3'])
one_hot_pay3 = np.array(one_hot_pay3)
one_hot_pay3 = np.hstack((one_hot_pay3 , np.zeros(10000).reshape(10000,1)))

'''#PAY4(6) 11
one_hot_pay4 = pd.get_dummies(table_x['PAY_4'])
one_hot_pay4 = np.array(one_hot_pay4)

#PAY5(7) 9
one_hot_pay5 = pd.get_dummies(table_x['PAY_5'])
one_hot_pay5 = np.array(one_hot_pay5)
one_hot_pay5 = np.hstack((one_hot_pay5[: , 0:3] , np.zeros(10000).reshape(10000,1) , one_hot_pay5[: , 3:] , np.zeros(10000).reshape(10000,1)))


#PAY6(8) 9
one_hot_pay6 = pd.get_dummies(table_x['PAY_6'])
one_hot_pay6 = np.array(one_hot_pay6)
one_hot_pay6 = np.hstack((one_hot_pay6[: , 0:3] , np.zeros(10000).reshape(10000,1) , one_hot_pay6[: , 3:] , np.zeros(10000).reshape(10000,1)))'''




x = np.hstack((bias , x_test[ : , 0:1] , one_hot_e , one_hot_m , one_hot_pay0 , one_hot_pay2 , one_hot_pay3 ,  x_test[ : , 9 :]))

w_file = np.load(sys.argv[2])
w = w_file[ 0:56 , :]
a = w_file[ 56: , :]
########################## normalization ######################################################
max = a[0]
min = a[1]
x[0:10000 , 1:2] = (x[0:10000 ,  1:2] - min) / (max - min)
for i in range(12):
    max = a[i*2 + 2]
    min = a[i*2 + 3]
    x[0:10000 , i+44:i+45] = (x[0:10000 ,  i+44:i+45] - min) / (max - min)
########################## testing process #################################################

y = 1 + np.exp(x.dot(w) * -1)
y = 1 / y

for i in range(10000):
    if y[i] > 0.469 : y[i] = 1
    else : y[i] = 0
y = y.astype(np.int)



########################## output ################################################
      
index = np.array([["id_" + str(i)] for i in range(10000)])
solution = np.hstack((index,y))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'Value']
solution.to_csv(sys.argv[3] , columns = ['id' , 'Value'] ,  index = False , sep = ',')






