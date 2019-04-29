import numpy as np
import keras
import sys

class PeopleSet:
    def __init__(self,file):
        self.id = []
        labelsArray = []
        tensorArray = []
        labeltmp = []
        if(len(file) == 892):
            offset = 0
        else:
            offset = 1
        for num, line in enumerate(file):
            if(num == 0):
                continue
            tmp = []
            
            inputs = line.split(',')
            if(inputs[0] == ""):
                break
            #Grab the passenger ID
            self.id.append(int(inputs[0]))

            #checks if the input file is test or training
            #if it is training set then add the labels
            if len(file) == 892:
                labeltmp.append(int(inputs[1-offset]))

            #Grab the pclass
            tmp.append(int(inputs[2-offset]))

            #Grab the gender
            gender = inputs[5-offset]
            if(gender == "male"):
                tmp.append(0)
            else:
                tmp.append(1)

            #This conditional handles missing age info,if it is missing then 0
            if(inputs[6-offset] == ""):
                tmp.append(0)
            else:
                tmp.append(float(inputs[6-offset]))

            #Grab the SibSp
            tmp.append(int(inputs[7-offset]))

            #Grab the Parch
            tmp.append(int(inputs[8-offset]))

            #Grab the Fare
            fare = inputs[10-offset]
            if(len(fare) == 0):
                fare = 0
            tmp.append(float(fare))

            #Grab embarked
            port_s = 1 if("S" in inputs[12-offset]) else 0
            if(len(inputs[12-offset]) == 1):
                port_s = 1
            tmp.append(int(port_s))
            port_q = 1 if("Q" in inputs[12-offset]) else 0
            tmp.append(int(port_q))
            port_c = 1 if("C" in inputs[12-offset]) else 0
            tmp.append(int(port_c))
            tensorArray.append(tmp)
        self.tensor = np.array(tensorArray, dtype=np.float32)
        self.labels = np.array(labeltmp)
        
    def normalize_samples(data):
        data -= data.mean(axis=0)
        data /= data.std(axis=0)
        return data

#Reads
f = open(str(sys.argv[1]),"r")
train_data = PeopleSet(f.readlines())
train_data.tensor = PeopleSet.normalize_samples(train_data.tensor)

from keras import models
from keras import layers

#Model implementation
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.tensor.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

#K-fold validation to handle small training set
k = 4
num_val_samples = len(train_data.id)//k
num_epochs = 3 
all_scores = []
for i in range(k):
    val_data = train_data.tensor[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_data.labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data.tensor[:i * num_val_samples],
         train_data.tensor[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_data.labels[:i * num_val_samples],
         train_data.labels[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)

    val_bce, val_acc = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_acc)
    
#Clear memory, build new PeopleSet class
from keras import backend as K
K.clear_session()
test_in = open(str(sys.stdin))
test_data = PeopleSet(test_in.readlines())
model = build_model()

model.fit(train_data.tensor,train_data.labels,
         epochs=2,batch_size=16,verbose=0)
labels = model.predict(test_data.tensor)
output = np.around(labels,0).astype(int)


sys.stdout.write("PassengerId,Survived\n")
for i in range(0,len(output)):
    sys.stdout.write(str(test_data.id[i])+","+str(output[i][0])+'\n')
    