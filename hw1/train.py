import pandas as pd
import numpy as np
import sys
import time 

start = time.time()
##################################### read file #####################################
csv_file = sys.argv[1]
table = pd.read_csv(csv_file , encoding = 'big5')
table1 = pd.DataFrame(table)
table1 = table1.drop([ i for i in table1.index if (i-9)%18 != 0 ])
table1 = table1.drop(columns = table1.columns[0:3])
table1.index = (table1.index - 9) // 18

table2 = pd.DataFrame(table)
table2 = table2.drop([ i for i in table2.index if (i-8)%18 != 0 ])
table2 = table2.drop(columns = table2.columns[0:3])
table2.index = (table2.index - 8) // 18

table3 = pd.DataFrame(table)
table3 = table3.drop([ i for i in table3.index if (i-2)%18 != 0 ])
table3 = table3.drop(columns = table3.columns[0:3])
table3.index = (table3.index - 2) // 18

table4 = pd.DataFrame(table)
table4 = table4.drop([ i for i in table4.index if (i-12)%18 != 0 ])
table4 = table4.drop(columns = table4.columns[0:3])
table4.index = (table4.index - 12) // 18

table5 = pd.DataFrame(table)
table5 = table5.drop([ i for i in table5.index if (i-5)%18 != 0 ])
table5 = table5.drop(columns = table5.columns[0:3])
table5.index = (table5.index - 5) // 18

table6 = pd.DataFrame(table)
table6 = table6.drop([ i for i in table6.index if (i-10)%18 != 0 ])
table6 = table6.drop(columns = table6.columns[0:3])
table6.index = (table6.index - 10) // 18

table7 = pd.DataFrame(table)
table7 = table7.drop([ i for i in table6.index if (i-7)%18 != 0 ])
table7 = table7.drop(columns = table6.columns[0:3])
table7.index = (table7.index - 7) // 18

############################### data processing ############################################
temp = np.array(table1).flatten()
temp = temp.astype(np.float)
index = []
x_25 = np.array([temp[i+j:i+j+10] for i in range(0,5281,480) for j in range(471)])
for i in range(5652) :
    if np.max(x_25[i]) > 150 or np.min(x_25[i]) < 0 : index.append(i)

temp3 = np.array(table2).flatten()
temp3 = temp3.astype(np.float)
x_10 = np.array([temp3[i+j:i+j+10] for i in range(0,5281,480) for j in range(471)])

temp4 = np.array(table3).flatten()
temp4 = temp4.astype(np.float)
x_co = np.array([temp4[i+j:i+j+10] for i in range(0,5281,480) for j in range(471)])
for i in range(5652) :
    if np.min(x_co[i]) < 0  : index.append(i)

for i in range(5652) :
    for j in range(10):
        if x_25[i][j] == 0 and x_10[i][j] == 0 and x_co[i][j] == 0 :
            index.append(i)
            break 

for i in range(5652):
    for j in range(10):
        if x_10[i][j] == 0 and j == 0 : x_10[i][j] == (x_10[i][1] + x_10[i][2])/2
        elif x_10[i][j] == 0 and j == 9 : x_10[i][j] == (x_10[i][7] + x_10[i][8])/2
        elif x_10[i][j] == 0 : x_10[i][j] == (x_10[i][j-1] + x_10[i][j+1])/2

temp5 = np.array(table4).flatten()
temp5 = temp5.astype(np.float)
x_so2 = np.array([temp5[i+j:i+j+10] for i in range(0,5281,480) for j in range(471)])
for i in range(5652) :
    if np.min(x_so2[i]) < 0  : index.append(i)

temp6 = np.array(table5).flatten()
temp6 = temp6.astype(np.float)
x_no2 = np.array([temp6[i+j:i+j+10] for i in range(0,5281,480) for j in range(471)])
for i in range(5652) :
    if np.min(x_no2[i]) < 0  : index.append(i)

temp7 = np.array(table6).flatten()
for i in range(5760):
    if temp7[i] == 'NR' : temp7[i] = '0'
temp7 = temp7.astype(np.float)

x_rainfall = np.array([temp7[i+j:i+j+10] for i in range(0,5281,480) for j in range(471)])
for i in range(5652) :
    if np.max(x_rainfall[i]) > 39 : index.append(i)

index = list(set(index))


x_25 = np.delete(x_25 , index , 0)
x_10 = np.delete(x_10 , index , 0)
x_co = np.delete(x_co , index , 0)
x_so2 = np.delete(x_so2 , index , 0)
x_no2 = np.delete(x_no2 , index , 0)
x_rainfall = np.delete(x_rainfall , index , 0)
y = np.array([[i[9]] for i in x_25])
x_25square = x_25**2
x_10square = x_10**2
############################################ feature processing ################################################
x_25 = x_25[0:5266 , 0:9]
x_10 = x_10[0:5266 , 1:9]
x_25square = x_25square[0:5266 , 3:9]
x_10square = x_10square[0:5266 , 3:9]
x_co = x_co[0:5266 , 3:9]
x_so2 = x_so2[0:5266 , 6:9]
x_no2 = x_no2[0:5266 , 8:9]
x_rainfall = x_rainfall[0:5266 , 7:9]

bb = np.array([[1] for i in range(5266)])
x_hat = np.hstack(( bb.astype(np.float) , x_25 , x_10 ,x_co , x_25square , x_10square , x_no2 , x_so2 ,  x_rainfall ))
print(x_hat.shape)
w = np.array([np.random.rand(1)*5 for i in range(42)])


ran = np.hstack((y,x_hat))
np.random.shuffle(ran)

x_validate = ran[4212:5266 , 1:43]
y_validate = ran[4212:5266 , 0:1]

x_hat = ran[0:4212, 1:43]
y = ran[0:4212, 0:1]

########################## normalization ######################################################
x_25_max = np.max(x_hat[0:4212 , 1:10])
x_25_min = np.min(x_hat[0:4212 , 1:10])
print(x_25_max , x_25_min)
x_hat[0:4212 , 1:10] = (x_hat[0:4212 , 1:10] - x_25_min) / (x_25_max - x_25_min)
x_validate[0:1054 , 1:10] = (x_validate[0:1054 , 1:10] - x_25_min) / (x_25_max - x_25_min)

x_10_max = np.max(x_hat[0:4212 , 10:18])
x_10_min = np.min(x_hat[0:4212 , 10:18])
print(x_10_max , x_10_min)
x_hat[0:4212 , 10:18] = (x_hat[0:4212 , 10:18] - x_10_min) / (x_10_max - x_10_min)
x_validate[0:1054 , 10:18] = (x_validate[0:1054 , 10:18] - x_10_min) / (x_10_max - x_10_min)

x_co_max = np.max(x_hat[0:4212 , 18:24])
x_co_min = np.min(x_hat[0:4212 , 18:24])
print(x_co_max , x_co_min)
x_hat[0:4212 , 18:24] = (x_hat[0:4212 , 18:24] - x_co_min) / (x_co_max - x_co_min)
x_validate[0:1054 , 18:24] = (x_validate[0:1054 , 18:24] - x_co_min) / (x_co_max - x_co_min)


x_25square_max = np.max(x_hat[0:4212 , 24:30])
x_25square_min = np.min(x_hat[0:4212 , 24:30])
print(x_25square_max , x_25square_min)
x_hat[0:4212 , 24:30] = (x_hat[0:4212 , 24:30] - x_25square_min) / (x_25square_max - x_25square_min)
x_validate[0:1054 , 24:30] = (x_validate[0:1054 , 24:30] - x_25square_min) / (x_25square_max - x_25square_min)

x_10square_max = np.max(x_hat[0:4212 , 30:36])
x_10square_min = np.min(x_hat[0:4212 , 30:36])
print(x_10square_max , x_10square_min)
x_hat[0:4212 , 30:36] = (x_hat[0:4212 , 30:36] - x_10square_min) / (x_10square_max - x_10square_min)
x_validate[0:1054 , 30:36] = (x_validate[0:1054 , 30:36] - x_10square_min) / (x_10square_max - x_10square_min)

x_so2_max = np.max(x_hat[0:4212 , 37:40])
x_so2_min = np.min(x_hat[0:4212 , 37:40])
print(x_so2_max , x_so2_min)
x_hat[0:4212 , 37:40] = (x_hat[0:4212 , 37:40] - x_so2_min) / (x_so2_max - x_so2_min)
x_validate[0:1054 , 37:40] = (x_validate[0:1054 , 37:40] - x_so2_min) / (x_so2_max - x_so2_min)


x_no2_max = np.max(x_hat[0:4212 , 36:37])
x_no2_min = np.min(x_hat[0:4212 , 36:37])
print(x_no2_max , x_no2_min)
x_hat[0:4212 , 36:37] = (x_hat[0:4212 , 36:37] - x_no2_min) / (x_no2_max - x_no2_min)
x_validate[0:1054 , 36:37] = (x_validate[0:1054 , 36:37] - x_no2_min) / (x_no2_max - x_no2_min)

x_rainfall_max = np.max(x_hat[0:4212 , 40:42])
x_rainfall_min = np.min(x_hat[0:4212 , 40:42])
print(x_rainfall_max , x_rainfall_min)
x_hat[0:4212 , 40:42] = (x_hat[0:4212 , 40:42] - x_rainfall_min) / (x_rainfall_max - x_rainfall_min)
x_validate[0:1054 , 40:42] = (x_validate[0:1054 , 40:42] - x_rainfall_min) / (x_rainfall_max - x_rainfall_min)


########################## training process #################################################

g_scalar = 1
l_rate = 1
accum = 0
count = 5000000
lamb = 0.01
loss_best = ((y_validate - x_validate.dot(w)).T.dot(y_validate - x_validate.dot(w))/1054)
w_best = w
while count > 0:
    gradient = 2*(x_hat.T.dot(x_hat).dot(w)) - 2*(x_hat.T.dot(y)) - 2*lamb*w
    g_scalar = gradient.T.dot(gradient)
    accum += g_scalar
    count -= 1
    w -= l_rate*gradient/(accum)**0.5
    
    if count%100 == 0 :
    	loss = (y_validate - x_validate.dot(w)).T.dot(y_validate - x_validate.dot(w))
    	loss /= 1054
    	if loss > loss_best : break
    	loss_best = loss
    	print(count , " " , loss**0.5)
########################## output ################################################

w = np.vstack((x_25_min , x_25_max , x_10_min ,x_10_max , x_co_min ,x_co_max ,  x_no2_min , x_no2_max ,x_25square_min , x_25square_max ,x_10square_min ,x_10square_max  , x_so2_min , x_so2_max , x_rainfall_min , x_rainfall_max ,  w))         

end = time.time()
t = end - start
h = t // 3600
t = t % 3600
m = t // 60 
s = (t % 60)//1
print("Run Time : ", h , " hrs " , m , " min " , s , " s  , loss = " , loss**0.5 , " count = " , count) 

f = open("./weight.np" , "wb")
np.save(f , w)
f.close()	