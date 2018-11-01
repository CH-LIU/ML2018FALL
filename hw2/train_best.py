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

train_y_file = sys.argv[2]
table_y = pd.read_csv(train_y_file , encoding = 'big5')
table_y = pd.DataFrame(table_y)


############################### data processing ############################################

x_train = np.array(table_x)
x_train = x_train.astype(np.float)
y_train = np.array(table_y)
y = y_train.astype(np.float)
bias = np.ones(20000).reshape(20000,1).astype(np.float)

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

#PAY2(4) 10 + 1 (8)
one_hot_pay2 = pd.get_dummies(table_x['PAY_2'])
one_hot_pay2 = np.array(one_hot_pay2)
one_hot_pay2 = np.hstack((one_hot_pay2 , np.zeros(20000).reshape(20000,1)))
#PAY3(5) 11
one_hot_pay3 = pd.get_dummies(table_x['PAY_3'])
one_hot_pay3 = np.array(one_hot_pay3)

'''#PAY4(6) 11
one_hot_pay4 = pd.get_dummies(table_x['PAY_4'])
one_hot_pay4 = np.array(one_hot_pay4)

#PAY5(7) 10 + 1 (1)
one_hot_pay5 = pd.get_dummies(table_x['PAY_5'])
one_hot_pay5 = np.array(one_hot_pay5)
one_hot_pay5 = np.hstack((one_hot_pay5[: , 0:3] , np.zeros(20000).reshape(20000,1) , one_hot_pay5[: , 3:]))

#PAY6(8) 10 + 1 (1)
one_hot_pay6 = pd.get_dummies(table_x['PAY_6'])
one_hot_pay6 = np.array(one_hot_pay6)
one_hot_pay6 = np.hstack((one_hot_pay6[: , 0:3] , np.zeros(20000).reshape(20000,1) , one_hot_pay6[: , 3:]))'''



x = np.hstack((bias , x_train[ : , 0:1] , one_hot_e , one_hot_m , one_hot_pay0 , one_hot_pay2 , one_hot_pay3 , x_train[ : , 9 :]))
print(x.shape)

w = np.array([np.random.rand(1) for i in range(56)])


########################## normalization ######################################################
max = np.max( x[0:20000 ,  1:2] )
min = np.min( x[0:20000 ,  1:2] )
x[0:20000 , 1:2] = (x[0:20000 ,  1:2] - min) / (max - min)
a = np.array([[max],[min]])
for i in range(44 , 56):
    max = np.max( x[0:20000 ,  i:i+1] )
    min = np.min( x[0:20000 ,  i:i+1] )
    x[0:20000 , i:i+1] = (x[0:20000 ,  i:i+1] - min) / (max - min)
    b = np.array([[max],[min]])
    a = np.vstack((a,b))
########################## training process #################################################

g_scalar = 1
l_rate = 0.0001
accum = 0
count = 1000000
lamb = 0.01
temp = 1+ np.exp(x.dot(w) * -1)
temp = 1 / temp
loss_best = -( y.T.dot(np.log(temp)) + (1 - y).T.dot(np.log((1-temp))) )
w_best = w
while count > 0:
    temp = 1+ np.exp(x.dot(w) * -1)
    temp = 1 / temp
    gradient = (-x.T.dot((y - temp)))
    #gradient = 2*(x_hat.T.dot(x_hat).dot(w)) - 2*(x_hat.T.dot(y)) - 2*lamb*w
    #g_scalar = gradient.T.dot(gradient)
    #accum += g_scalar
    count -= 1
    l_rate *= 0.9999
   # w -= l_rate*gradient/(accum)**0.5 
    w -= l_rate*gradient
    if count % 100 == 0 :
        #print(temp)
        loss = -( y.T.dot(np.log(temp)) + (1 - y).T.dot(np.log((1-temp))) )
        if loss > loss_best : break
        w_best = w
        loss_best = loss
        print(count , " " , loss)
########################## output ################################################

w = np.vstack((w , a))
f = open("./w/k_hot_4pay.np" , "wb")
np.save(f , w)
f.close()