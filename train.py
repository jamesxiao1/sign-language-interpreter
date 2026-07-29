import numpy as np
import tensorflow as tf 
import sys 
import csv 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2, # hold 20% for testing,80% for training 
    stratify=y, # piles keep the same proportion of each letter 
    random_state=42 # random number, for reproducible results
)

# print("train:", x_train.shape, y_train.shape)
# print("test:", x_test.shape, y_test.shape)
# print("train labels:", np.bincount(y_train)) 
# print("test labels:", np.bincount(y_test))

model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)), # define shape of input data,each training example has 63 features
    tf.keras.layers.Dense(64,activation="relu"), # dense is fully connected layer, 64 is 64 neurons, each neuron recieves info from all 63 features
    tf.keras.layers.Dense(32,activation="relu"), # 32 neurons, 64 inputs each 
    tf.keras.layers.Dense(3,activation="softmax") # 3 neurons for a,b,l

    # input width =feature count(63), output width=class count(3)
])

model.compile( # configures settings before training?
    optimizer="adam", # smart nudger, gradient descent
    loss="sparse_categorical_crossentropy",#wrongness score
    metrics=["accuracy"]
)

model.fit( #guess,measure,blame,nudge loop
    x_train,y_train, 
    validation_data=(x_test,y_test),
    epochs=50,# loops 
    batch_size=16 #process 16 examples ata time before each nudge, faster than one at a time 
)

"""
loss -wrongness in data,show drop close to 0 as epoch inc 
accuracy-% correct in data, should rise toward 1.0
val_loss/val_accuracy-same as before but for held-back test set, val accuracy is important, should rise to 1.0
"""

data_file.close()
model.save("sign_model.keras")
np.save("label_classes.npy",encoder.classes_)
print("saved model and labels")