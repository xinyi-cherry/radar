import pandas as pd
import json

df = pd.read_csv('data_train.csv')
l = df['dec_outputs']
s = set(l.to_list())

with open('dict.json','r') as h:
   d = json.load(h) 

labels = []

for str in s:
    num = []
    split = str.split(' ')
    for sub in split:
        num.append(d.index(sub))
    labels.append(num)

with open('output.json','w') as h:
    json.dump(labels,h)