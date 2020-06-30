import cv2 as cv
import numpy as np
import os
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score


# dataset avilable here: https://www.kaggle.com/paultimothymooney/blood-cells

trainData="C:\\Users\\n0745509\\Downloads\\blood-cells\\dataset2-master\\dataset2-master\\images\\TRAIN"
testData="C:\\Users\\n0745509\\Downloads\\blood-cells\\dataset2-master\\dataset2-master\\images\\TEST"


classLabels=["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]

# storing img and label as a pair
train_data=[]

for category in classLabels:
    
        label=classLabels.index(category)
        path=os.path.join(trainData,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1) # returns loaded image unchanged 
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img=cv.resize(img,(60,60)) # resize for quick training time and less demand on memory            
            train_data.append([img,label]) # adds the image and the index name to the list

test_data=[]

for category in classLabels:
       
        
        label=classLabels.index(category)
        path=os.path.join(testData,category)
        
        for img_file in os.listdir(path):
            
            img=cv.imread(os.path.join(path,img_file),1)
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img=cv.resize(img,(60,60)) 
            test_data.append([img,label])




#shuffle the dataset fo good result
random.shuffle(train_data)
random.shuffle(test_data)



X_train=[] # features go here (images)
y_train=[] # labels go here (class labels)

#split features and labels into x test and y test - used for training model 
for features,label in train_data:       
    X_train.append(features)
    y_train.append(label)



X_test=[] # features go here (images)
y_test=[] # labels go here (class labels)


#split features and labels into x test and y test - used for testing model 
for features,label in test_data:
    X_test.append(features)
    y_test.append(label)


    


X_train=np.array(X_train).reshape(-1,60,60,3) 
X_train=X_train/255.0 # close value to 0 - convert to greyscale

X_test=np.array(X_test).reshape(-1,60,60,3)
X_test=X_test/255.0 # close value to 0 - convert to greyscale


# encode labels into numveric values
# used for categorical_crossentropy
encodeTrain=to_categorical(y_train) 
encodeTest=to_categorical(y_test) 





model=Sequential()

# 3 x convo layers
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(60,60,3)))
model.add(MaxPooling2D(pool_size=(2,2))) # max pool select argmax value 
model.add(Dropout(0.20))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.20))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.40))

# flattens input into 1D
model.add(Flatten())

#fully-connected layers
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))

#final layer - output that represents the probability distributions of a list of potential outcomes (that being the class labels)         
model.add(Dense(4,activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#split the 20% train dataset for validation, batch size set to help with memory demand 
model.fit(X_train,encodeTrain,epochs=25,batch_size=128,validation_split=0.2)

test_loss,test_acc=model.evaluate(X_test,encodeTest)


# conduct testing of trained model using testing data 
y_pred=model.predict_classes(X_test)


model.save('blood_cell_model.h5')


# code used to display visual of training accuracy against rest accuracy for each epoch 
#####################################

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper right')
plt.show()
#######################################



# test model with random individual samples between 1 and 100
for i in range(15):
        x = random.randint(1, 100)
        print("Actual=%s, Predicted=%s" % (y_test[x], y_pred[x]))

#accuracy_score based on test predications 
print(accuracy_score(y_test,y_pred))

