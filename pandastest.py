import pandas as pd

l = [] #空列表
for i in range(1,78): #循环第一个编号
    for j in range(1,11): #1-10循环10次
        filename = 'data{}_{}_Raw_0_'.format(i,j)
        print(filename)
        
pd.read_excel('')

print(pd.DataFrame(l))