# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:25:36 2020

@author: OmarL
"""



import matplotlib.pyplot as plt

import tensorflow as tf
import keras
import random
import numpy as np
import gym
from gym import wrappers



FLOOR = {
        
    "ecg":['SFFHFFGF',
              'HFHFFHFH',
              'HFFFHFFH',
              'HHHFHFHH',
              'HFHFFFFF'],
              
    "exit":['SFFHFFF',
              'FHFFFHF',
              'FFFHHFF',
              'HHFFFHF',
              'FFHFFFG'],
              
    "reception": ['SFHFFH',
               'FFFHFF',
               'HFFFFF',
               'FGFFHH'],
               
    "toilet": ['SHFH',
               'FFGF',
               'HFFH']
    }




  

user = input("enter loc: ")   


if user in FLOOR:
    
    #desc = FLOOR[temp[x]]
    #desc = np.asarray(desc,dtype='c')
    env = gym.make('FrozenLake-v0', desc=generate_random_map(8, 0))
    
    env.render()
    
    
    
  
else:
    raise Exception ("error! invlaid floor plan. Please select one of the following: {}".format("floor0, floor1, floor2, or floor3"))
    



count_actions = env.action_space.n
count_states = env.observation_space.n



from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam

ENV_NAME = "Taxi-v3"

env = gym.make(ENV_NAME)
env.render()
np.random.seed(123)
env.seed(123)

env.reset()



count_actions = env.action_space.n
count_states = env.observation_space.n




model = Sequential()
model.add(Embedding(500, 10, input_length=1))
model.add(Reshape((10,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(count_actions, activation='linear'))



###############################################################################################
import pandas as pd
import numpy as np
import os


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from numpy.testing import assert_allclose



"""
REFERENCE: https://www.kaggle.com/harishreddy18/english-to-french-translation

changes i made:
    *added auto save model at every epoch to save model if its accuracy is better than previous epoch 
    *added embedding layer  
    *changed hyperparamters: number of filters, learning rate etc.
    *added LSTM layer to avoid vanishing gradients problems
    *saved preprocessing function to file so that i can load data into varialbes without having to re-run algorithm (saves time)
    *added dropout layer to help fight against overfitting as this was a problem i was having. 
    

"""

# load data function
def tokenize(x):
    
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

# makes sure that both french sentences and english sentences have the same length by adding padding at the end
def pad(x, length=None):
    
    sentencePadding = pad_sequences(x, maxlen = length, padding = 'post')
    return sentencePadding

# get final form; french translation
def logits_to_text(logits, tokenizer):

    idx_to_words = {id: word for word, id in tokenizer.word_index.items()}
    idx_to_words[0] = ""

    return ' '.join([idx_to_words[prediction] for prediction in np.argmax(logits, 1)])
                 

def preprocess(x, y):
    
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

def load_data(path):

    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


def rnnModel(X, y, en_vocab_size, fr_voab_size, frLength):
    
    # hyperparameters
    lr = 0.001
    batch_size = 32
    epochs = 5
    validation_split = 0.1
    verbose = 1
    

    
    model = Sequential()
    model.add(Embedding(en_vocab_size+1,
                        256, 
                        input_length=frLength,
                        input_shape=(frLength,)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(Dense(fr_voab_size+1, activation="softmax"))
    

    model.compile(optimizer=Adam(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    # auto saves model if validation accuracy is greater than previous epoch
    checkpoint = ModelCheckpoint("new_eng-fr_model.h5",
                                 monitor='val_acc', 
                                 verbose=verbose,
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint] 
    
    
    model.fit(X,
              y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_split=validation_split,
              callbacks=callbacks_list)
    
    
    new_model = load_model("new_eng-fr_model")
    assert_allclose(model.predict(X),
                    new_model.predict(X),
                    lr)
    
    checkpoint = ModelCheckpoint("new_eng-fr_model",
                                 monitor='val_acc',
                                 verbose=verbose,
                                 save_best_only=True, 
                                 mode='max')
    callbacks_list = [checkpoint]
    
    new_model.fit(X,
                  y,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=validation_split,
                  callbacks=callbacks_list)

    
    
    
# samll vocab dataset
enPath = "small_vocab_en.csv"
frPath = "small_vocab_fr.csv"


# much larger vocab dataset: takes too long to preprocess on my computer
#enPath = "C:\\Users\\OmarL\\Downloads\\fr-en\\en.csv"
#frPath = "C:\\Users\\OmarL\\Downloads\\fr-en\\fr.csv"

# load dataset
enData = load_data(enPath)
frData = load_data(frPath)

# view sanple of data by slicing
#print(enData[:2])
#print("\n")
#print(frData[:2])




# ***IMPORTANT*** to note for model parameters to avoid out of index runtime error 
# english length = 15
# french length = 21 
# english voc size = 213
# french voc size = 345

preprocessEnData, preprocessFrData, enTokenize, frTokenize = preprocess(enData, frData)

import pickle 

with open("largeFile.pkl", 'wb') as f:
    pickle.dump([preprocessEnData, preprocessFrData, enTokenize, frTokenize], f,protocol=4)


englishLength = preprocessEnData.shape[1]
frenchLength = preprocessFrData.shape[1]

enVocSize = len(enTokenize.word_index)
frVocSize = len(frTokenize.word_index)

print(englishLength, frenchLength)
print(enVocSize, frVocSize)

 
X = pad(preprocessEnData, frenchLength)
y = preprocessFrData

rnnModel(X, y, enVocSize, frVocSize, frenchLength)
    

### TESTING ### 
test_sentence = 'he saw a old yellow truck'

model = load_model("eng-fr_model.h5")

preprocessEnData, preprocessFrData, enTokenize, frTokenize = preprocess(enData, frData)


sentence = [enTokenize.word_index.get(word) for word in test_sentence.split()]
print(sentence)
sentence = pad_sequences([sentence], maxlen=21, padding='post')
print("French Translation: ", logits_to_text(model.predict(sentence)[0], frTokenize))







