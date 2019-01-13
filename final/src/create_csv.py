import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_df = pd.read_csv("train_old.csv")
more_df = pd.read_csv("./HPAv18RBGY_wodpl.csv")

new_df = train_df
#new_df = pd.DataFrame()
lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27,16,13,12,22,18,27,15,1,9,9,10,3,4,6,10]
for i in lows:
    target = str(i)
    indicies = train_df.loc[train_df['Target'] == target].index
    new_df = pd.concat([new_df,train_df.loc[indicies]], ignore_index=True)
    indicies = train_df.loc[train_df['Target'].str.startswith(target+" ")].index
    new_df = pd.concat([new_df,train_df.loc[indicies]], ignore_index=True)
    indicies = train_df.loc[train_df['Target'].str.endswith(" "+target)].index
    new_df = pd.concat([new_df,train_df.loc[indicies]], ignore_index=True)
    indicies = train_df.loc[train_df['Target'].str.contains(" "+target+" ")].index
    new_df = pd.concat([new_df,train_df.loc[indicies]], ignore_index=True)

#lows = [15,9,10,26,20,24,17,8,27]
for i in lows:
    target = str(i)
    indicies = more_df.loc[more_df['Target'] == target].index
    new_df = pd.concat([new_df,more_df.loc[indicies]], ignore_index=True)
    indicies = more_df.loc[more_df['Target'].str.startswith(target+" ")].index
    new_df = pd.concat([new_df,more_df.loc[indicies]], ignore_index=True)
    indicies = more_df.loc[more_df['Target'].str.endswith(" "+target)].index
    new_df = pd.concat([new_df,more_df.loc[indicies]], ignore_index=True)
    indicies = more_df.loc[more_df['Target'].str.contains(" "+target+" ")].index
    new_df = pd.concat([new_df,more_df.loc[indicies]], ignore_index=True)


row_25 = new_df.loc[new_df['Target'] == '25']
row_0 = new_df.loc[new_df['Target'] == '0']
index_25 = new_df.loc[new_df['Target'] == '25'].index    #1470
index_0 = new_df.loc[new_df['Target'] == '0'].index      #2414
row_0 = shuffle(row_0)
row_25 = shuffle(row_25)
new_df = pd.concat([new_df,row_25.iloc[0:735]], ignore_index=True)
new_df = pd.concat([new_df,row_0.iloc[0:1207]], ignore_index=True)
#print(new_df)
#new_df = new_df.drop(list(index_0))
#new_df = new_df.drop(list(index_25))






new_df.to_csv("train_25_0.csv" , index = False)
#new_train_df, new_test_df = train_test_split(more_df,test_size = 0.06,random_state = 2050)
#new_test_df.to_csv("report.csv")
