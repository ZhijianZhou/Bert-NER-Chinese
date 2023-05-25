import pandas as pd
import csv
df2 = pd.read_csv('test.csv')
count = 0
for i in range(0,df2.shape[0]):
    if count != 0:
        count -= 1
        continue
    if df2.iloc[i,1] != "O":
        houzhui = df2.iloc[i,1]
        j = 1
        while(df2.iloc[i+j,1] == houzhui):
             j+=1
             count += 1
        if count == 0:
           df2.iloc[i,1] = "S-"+houzhui
        elif count == 1:
           df2.iloc[i,1] = "B-"+houzhui
           df2.iloc[i+1,1] = "E-"+houzhui
        else:
           df2.iloc[i,1] = "B-"+houzhui
           df2.iloc[i+count,1] = "E-"+houzhui
           for j in range(0,count-1):
                df2.iloc[i+j+1,1] = "M-"+houzhui
    
with open("test_plus.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id","expected"])
        for i in range(0,df2.shape[0]):
             data = df2.iloc[i,:]
             writer.writerow(data)