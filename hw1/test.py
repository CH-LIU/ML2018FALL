import pandas as pd
import numpy as np
import sys

##################################### read file #####################################

csv_file = sys.argv[1]
table = pd.read_csv(csv_file , encoding = 'big5' , header = None)
table1 = pd.DataFrame(table)
table1 = table1.drop([ i for i in table1.index if (i-9)%18 != 0 ])
table1 = table1.drop(columns = table1.columns[0:2])
table1.index = (table1.index - 9) // 18

table2 = pd.DataFrame(table)
table2 = table2.drop([ i for i in table2.index if (i-8)%18 != 0 ])
table2 = table2.drop(columns = table2.columns[0:2])
table2.index = (table2.index - 8) // 18

table3 = pd.DataFrame(table)
table3 = table3.drop([ i for i in table3.index if (i-2)%18 != 0 ])
table3 = table3.drop(columns = table3.columns[0:2])
table3.index = (table3.index - 2) // 18

table4 = pd.DataFrame(table)
table4 = table4.drop([ i for i in table4.index if (i-12)%18 != 0 ])
table4 = table4.drop(columns = table4.columns[0:2])
table4.index = (table4.index - 12) // 18

table5 = pd.DataFrame(table)
table5 = table5.drop([ i for i in table5.index if (i-12)%18 != 0 ])
table5 = table5.drop(columns = table5.columns[0:2])
table5.index = (table5.index - 12) // 18

table6 = pd.DataFrame(table)
table6 = table6.drop([ i for i in table6.index if (i-10)%18 != 0 ])
table6 = table6.drop(columns = table6.columns[0:2])
table6.index = (table6.index - 10) // 18

############################### data processing ############################################

x_25 = np.array(table1).reshape((260,9))
x_25 = x_25.astype(np.float)
for i in range(260) : 
    for j in range(9):
    	if x_25[i][j] < 0 : x_25[i][j] = 0
    	'''if x_25[i][j] == 0 and j == 0 : x_25[i][j] = (x_25[i][1] + x_25[i][2])/2
    	elif x_25[i][j] == 0 and j == 8 : x_25[i][j] =  (x_25[i][7] + x_25[i][6])/2
    	elif x_25[i][j] == 0 : x_25[i][j] =  (x_25[i][j-1] + x_25[i][j+1])/2 ''' 
x_10 = np.array(table2).reshape((260,9))
x_10 = x_10.astype(np.float)

for i in range(260):
	for j in range(9):
		if x_10[i][j] == 0 and j == 0 : x_10[i][j] == (x_10[i][1] + x_10[i][2])/2
		elif x_10[i][j] == 0 and j == 8 : x_10[i][j] == (x_10[i][6] + x_10[i][7])/2
		elif x_10[i][j] == 0 : x_10[i][j] == (x_10[i][j-1] + x_10[i][j+1])/2

x_co = np.array(table3).flatten()
x_co = x_co.astype(np.float)
for i in range(2340) :
    if x_co[i] < 0  : x_co[i] = 0
x_co = x_co.reshape((260,9))
'''for i in range(260):
    for j in range(9):
        if x_co[i][j] < 0 and j == 0 : x_co[i][j] == (x_co[i][1] + x_co[i][2])/2
        elif x_co[i][j] < 0 and j == 8 : x_co[i][j] == (x_co[i][6] + x_co[i][7])/2
        elif x_co[i][j] < 0 : x_co[i][j] == (x_co[i][j-1] + x_co[i][j+1])/2'''

x_so2 = np.array(table4).flatten()
x_so2 = x_so2.astype(np.float)
for i in range(2340):
    if x_so2[i] < 0 : x_so2[i] = 0
x_so2 = x_so2.reshape((260,9))
'''for i in range(260):
    for j in range(9):
        if x_so2[i][j] < 0 and j == 0 : x_so2[i][j] == (x_so2[i][1] + x_so2[i][2])/2
        elif x_so2[i][j] < 0 and j == 8 : x_so2[i][j] == (x_so2[i][6] + x_so2[i][7])/2
        elif x_so2[i][j] < 0 : x_so2[i][j] == (x_so2[i][j-1] + x_so2[i][j+1])/2'''


x_no2 = np.array(table5).flatten()
x_no2 = x_no2.astype(np.float)
for i in range(2340):
    if x_no2[i] < 0 : x_no2[i] = 0
x_no2 = x_no2.reshape((260,9))
'''for i in range(260):
    for j in range(9):
        if x_no2[i][j] < 0 and j == 0 : x_no2[i][j] == (x_no2[i][1] + x_no2[i][2])/2
        elif x_no2[i][j] < 0 and j == 8 : x_no2[i][j] == (x_no2[i][6] + x_no2[i][7])/2
        elif x_no2[i][j] < 0 : x_no2[i][j] == (x_no2[i][j-1] + x_no2[i][j+1])/2'''


temp7 = np.array(table6).flatten()
for i in range(2340):
    if temp7[i] == 'NR' : temp7[i] = '0'
    #else : temp7[i] = '1'
temp7 = temp7.astype(np.float)
'''for i in range(2340):
    if temp7[i] > 25 : temp7[i]  = 25'''
x_rainfall = temp7.reshape((260,9))

x_25square = x_25**2
x_10square = x_10**2

########################## normalization ######################################################
w = np.load(sys.argv[2])


x_25_max = w[1]
x_25_min = w[0]
x_25 = (x_25 - x_25_min) / (x_25_max - x_25_min)

x_10_max = w[3]
x_10_min = w[2]
x_10 = (x_10 - x_10_min) / (x_10_max - x_10_min)

x_co_max = w[5]
x_co_min = w[4]
x_co = (x_co - x_co_min) / (x_co_max - x_co_min)

x_so2_max = w[13]
x_so2_min = w[12]
x_so2 = (x_so2 - x_so2_min) / (x_so2_max - x_so2_min)

x_no2_max = w[7]
x_no2_min = w[6]
x_no2 = (x_no2 - x_no2_min) / (x_no2_max - x_no2_min)

x_25square_max = w[9]
x_25square_min = w[8]
x_25square = (x_25square - x_25square_min) / (x_25square_max - x_25square_min)

x_10square_max = w[11]
x_10square_min = w[10]
x_10square = (x_10square - x_10square_min) / (x_10square_max - x_10square_min)

x_rainfall_max = w[15]
x_rainfall_min = w[14]
x_rainfall = (x_rainfall - x_rainfall_min) / (x_rainfall_max - x_rainfall_min)

w = w[16 : ]

############################################ feature processing ################################################

x_so2 = x_so2[0:260 ,6:9]
x_co = x_co[0:260 , 3:9]
x_10 = x_10[0:260 , 1:9]
x_no2 = x_no2[0:260 , 8:9]
x_25square = x_25square[0:260 , 3:9]
x_10square = x_10square[0:260 , 3:9]
x_rainfall = x_rainfall[0:260 , 7:9]

bb = np.array([[1] for i in range(260)])
x_hat = np.hstack(( bb.astype(np.float) ,x_25 , x_10 ,x_co , x_25square , x_10square ,x_no2, x_so2 ,x_rainfall ))
y = x_hat.dot(w)

index = np.array([["id_" + str(i)] for i in range(260)])
solution = np.hstack((index,y))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'value']
solution.to_csv(sys.argv[3] , columns = ['id' , 'value'] ,  index = False , sep = ',')