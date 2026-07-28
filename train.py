import numpy as np
import tensorflow as tf 
import sys 
import csv 
from sklearn.preprocessing import LabelEncoder

def normalize_hand(row):
    # a row is 63 nums for one hand
    points=row.reshape(21,3) # make double matrix 21 rows of 3
    points=points-points[0] # no wrist
    size=np.linalg.norm(points,axis=1).max() # normalize
    points=points/size # finish normalize
    return points.reshape(-1) # put it back to 63 flat

labels=[]
features=[]
data_file=open("data.csv","r")
reader=csv.reader(data_file)
for row in reader: 
    labels.append(row[0])
    a_row=[]
    for val in row[1:]:
        a_row.append(float(val))
    features.append(a_row)

y=np.array(labels)
x=np.array(features)

# print("x shape",x.shape)
# print("y shape",y.shape)
# print("first",y[0])
# print("first len",len(x[0]))

encoder=LabelEncoder()
y=encoder.fit_transform(y) # fit=assign unique labels an integer, transform=convert y array from letters to integers,['a','a','b',...]->[0,0,1...]
# reverse later

# position and size don't matter 
normalized=[]
for row in x: 
    normalized.append(normalize_hand(row))
x=np.array(normalized)

# print("x shape",x.shape)
# print("first piont of row 0",x[0][:3])




data_file.close()