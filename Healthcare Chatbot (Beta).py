import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string
import wikipediaapi
import urllib as url
import re
from pprint import pprint
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, Policy
from rl.memory import SequentialMemory
import gym
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import pickle

import warnings
warnings.filterwarnings("ignore")

wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

pageURL = "https://api.nhs.uk/conditions/" #NHS url
subscriptionKey = "7cdb41e5ec594aa89f0068b7d17642ec" # generated sub key
request_headers = { 
  "subscription-key": subscriptionKey,
  "Accept": "application/json",
  "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0"
}


file = open('QuestionAnswerPair.txt','r', errors = "ignore")  #reads data from file   
 
raw = file.read()
raw = raw.lower()

## for bag of words function
#nltk.download('punkt') 
#nltk.download('wordnet') 
#nltk.download('stopwords')

## for natural language processing
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


lemmer = nltk.stem.WordNetLemmatizer()
    
####################### FUNCTIONS #######################

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(userInput):
    chatbotOutput=''
    sent_tokens.append(userInput)
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') #removes useless words
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        chatbotOutput=chatbotOutput+"I am sorry! I don't understand you"
        return chatbotOutput
    else:
        chatbotOutput = chatbotOutput+sent_tokens[idx+1]
        return chatbotOutput
    

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});') #contributes to cleaning output 
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


# plots selected image with the predication accuracy made by the model 
def plot_image(i, predictions_array, true_label, img):
              predictions_array, true_label, img = predictions_array, true_label[i], img[i]
              
              plt.grid(False)
              plt.xticks([]) # empty
              plt.yticks([]) # empty 
              plt.imshow(img, cmap=plt.cm.binary) # displays img from function parameter 
              
              predicted_label = np.argmax(predictions_array) 
              
              plt.xlabel("{} {:2.0f}% ".format(classLabels[predicted_label],
                                            100*np.max(predictions_array))) # percentage label = argmax value 
                                            
# plots graph showing value of each label - highlighting the argmax label                  
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label= predictions_array, true_label[i]
    
    plt.grid(False)
    plt.xticks(range(4), classLabels) # x label = all class labels 
    plt.ylabel("Accuracy")
    plt.title("Model Result")
    thisplot = plt.bar(range(4), predictions_array, color="#777777") # bar chart plotted using predication value of each class label
    plt.ylim([0, 1])
    
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red') # argmax label set to red for a clearer visual of largest value



# clean parameters function    
def errorHandling():
    params[1] = params[1].replace(" ", "")
    params[2] = params[2].replace(" ","")
    params[1] = ''.join(e for e in params[1] if e.isalnum())
    params[2] = ''.join(e for e in params[2] if e.isalnum())
    
    
def rlModelEmergencyRoute(actions, states, env):
    # hyperparameters
    lr = 0.001
    nb_episdoes=1
    
    model = Sequential()
    model.add(Embedding(states, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    
    model.load_weights("dqn_Emergency_Route_weights.h5f")
    
    memory = SequentialMemory(limit=10, window_length=1)
    
    policy = EpsGreedyQPolicy()
    
    dqn = DQNAgent(model=model,
                   nb_actions=actions,
                   memory=memory,
                   nb_steps_warmup=500,
                   target_model_update=0.002,
                   policy=policy)
    
   
    dqn.compile(Adam(lr=lr), 
                metrics=['mae'])
    
    dqn.test(env,
             nb_episodes=nb_episdoes,
             visualize=True)


def rlModelFloorRoute(actions, states, env, weights):
    
    # hyperparameters
    batch_size=32
    lr=0.001
    nb_episodes=1
    
    model = Sequential()
    model.add(Embedding(states, actions, input_length=1))
    model.add(Reshape((actions,)))
    
    model.load_weights(weights)
    
    memory = SequentialMemory(limit=10, window_length=1)
    
    policy = EpsGreedyQPolicy()
    
    
    dqn = DQNAgent(model=model,
                   nb_actions=actions,
                   memory=memory,
                   nb_steps_warmup=500,
                   target_model_update=0.002,
                   policy=policy,
                   enable_double_dqn=False,
                   batch_size=batch_size
                   )
    dqn.compile(Adam(lr=lr))
    
   
  
    dqn.test(env,
             nb_episodes=nb_episodes,
             visualize=True)
    
    
    
# function to translate sentence into french in reponse to the model predictions
def logits_to_text(logits, tokenizer):

    idx_to_words = {id: word for word, id in tokenizer.word_index.items()}
    idx_to_words[0] = ""

    return ' '.join([idx_to_words[prediction] for prediction in np.argmax(logits, 1)])
                 
    
################################################
    
    
    
    
  
import pandas as pd
filename = "ValuationDatabase.txt"
df = pd.read_csv(filename,delimiter="\n", dtype=str) # store text file into dataframe for easy formatting

                       
v = df.to_string(index=False) # remove index column from dataframe and convert it to string

folval = nltk.Valuation.fromstring(str(v)) # read string into vaulation, str just to make sure its a string
grammar_file = 'simple-sem.fcfg'
objectCounter = 0


import aiml
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="Rule-based Commands.xml")
string = ("Welcome please ask me what symptoms you're suffering from. I'll try my best to provide you",
      "with an accurate diagnosis.")
pprint(string)
reset = 'Y'
while True:
    #get user input
    try:
        
        userInput = input(">").upper()
       
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
  
    responseAgent = 'aiml'

    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            informationLoc = input("wiki or nhs \n").lower() #chose between wiki or nhs api
            wpage = wiki_wiki.page(params[1])        
             
            if  informationLoc == "wiki" and  wpage.exists():
                print(wpage.summary)
                print("Learn more at", wpage.canonicalurl)
       
            elif informationLoc == "nhs":
               
                request = url.request.Request(pageURL+params[1], headers=request_headers)
                contents = url.request.urlopen(request).read()
                stringContent = contents.decode("utf-8")
                
                pprint(cleanhtml(stringContent))

            else:
                print("Sorry, I don't know what that is.")
            cmd=00
                            
                    
###################################################################  
                    
        elif cmd==2:
           
            classLabels=["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]
        
            model = load_model('blood_cell_model.h5') # load pre_trained model
      
            root = tk.Tk()
            root.withdraw() # hides root to avoid crash
            file_path = filedialog.askopenfilename() # opens file directory 
            image = Image.open(file_path) # stores selected image into variable
            image = image.resize((60, 60)) #resize image to fit in model
            
            # convert to numpy array (#-1 means same size, 60x60 = height and width, 3 for R+G+B)
            image = np.array(image, dtype = 'float32').reshape(-1, 60, 60, 3) 
            image/=255.
      
            predictions = model.predict(image)
            
            #print(model.summary())

            # plots the selected input image and argmax value and its corresponding label - calls function
            #plt.subplot(1, 2, 1)
            plt.figure(figsize  = (10,5))
            plot_image(-1, predictions[-1], classLabels, image) # function called - always elected last image from array
            # plots all models predicaitons for each label in form of a graph - calls function

            plt.figure(figsize  = (10,5))
            plot_value_array(-1, predictions[-1], classLabels) # function called - always elected last image from array
            plt.show()
            plt.clr()
            
            # if statement for displaying largest value using argma, prints its corresponding label 
            if np.argmax(predictions) == 0:
                print("Image is: EOSINOPHIL")
            elif np.argmax(predictions) == 1:
                print("Image is: LYMPHOCYTE")
            elif np.argmax(predictions) == 2:
                 print("Image is: MONOCYTE")
            elif np.argmax(predictions) == 3:
                 print("Image is: NEUTROPHIL")
                     
            cmd=99
###################################################################

                
        elif cmd == 3: # CAN I TAKE * WITH *
             errorHandling() # run parameter cleaning function
             params[1] = params[1].lower() # convert parameter to lowercase
             params[2] = params[2].lower()
             if params[1] in folval: # check if parameter is in vaulation file 
                if params[2] in folval:
                     test = nltk.word_tokenize(v) # tokenize vaulation string to pick out important token
                     for x in range(len(test)):
                         if params[1] and params[2] == test[x]:
                             if (test[x+4], '') in folval['with']:
                                 print("Yes. You can take" + params[1] + " with " + params[2] )
                             else:
                                 print("No. You cannot take " + params[1] + " with " + params[2])
                else:
                    raise Exception("File location does not appear in the vaulation file: {}".format(params[2]))

             else:
                raise Exception("Medication name does not appear in the vaulation file: {}".format(params[1]))
                
             cmd=99
            
        elif cmd == 4: #IS * IN *
             errorHandling()
             params[1] = params[1].lower()
             params[2] = params[2].lower()

            
             if params[1] in folval:
                if params[2] in folval:
                     g = nltk.Assignment(folval.domain)
                     m = nltk.Model(folval.domain, folval)
                     sent = ' some ' + params[1] + ' is_in ' + params[2]
                     results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]

                     if results[2] == True:
                         print("Yes." + params[1] + " is in " + params[2])
                     else:
                         print("No.")
                else:
                    raise Exception("File location does not appear in the vaulation file: {}".format(params[2]))

             else:
                raise Exception("Medication name does not appear in the vaulation file: {}".format(params[1]))

                         
             cmd=99
            
        elif cmd == 5: # CAN I SAVE * IN *
            errorHandling()
            
            params[1] = params[1].lower()
            params[2] = params[2].lower()
           
            if params[1] in folval:
                if params[2] in folval:
                    o = 'o' + str(objectCounter)
                    objectCounter += 1
                    folval['o' + o] = o #insert constant
                    if len(folval[params[1]]) == 1: #clean up if necessary
                        if ('',) in folval[params[1]]:
                            folval[params[1]].clear()
                    folval[params[1]].add((o,)) #insert type of plant information
            
                    if len(folval["storing"]) == 1: #clean up if necessary
                        if ('',) in folval["storing"]:
                            folval["storing"].clear()
                    folval["storing"].add((o, folval[params[2]])) #insert location
    

                    if (folval["storing"]):
                 
                        print(params[1] + " has been saved to " + params[2])
                    else:
                        print("input not saved")
                else:
                    raise Exception("Medication name does not appear in the vaulation file: {}".format(params[2]))

            else:
                raise Exception("Medication name does not appear in the vaulation file: {}".format(params[1]))
    
            cmd=99
            
        elif cmd == 6: # DOES * REQUIRE A PRESCRIPTION
            params[1] = params[1].replace(" ", "")
            params[1] = ''.join(e for e in params[1] if e.isalnum())
            params[1] = params[1].lower()

            if params[1] in folval:
         
                tokenV = nltk.word_tokenize(v)
                flag = True
                for x in range(len(tokenV)):
                    if params[1]== tokenV[x]:

                        if (tokenV[x+4], '') in folval['commonMeds']:
                            print("No. " + params[1] + " can be found in many hosueholds and"+
                                  "are widely available in standard supermarkets")
                            flag = True
                            break      
                    else:
                        flag = False         
                if flag == False:
                    print("Yes. " + params[1].lower() + " requires a prescription from your local healthcare unit")
            else:
                raise Exception("Medication name does not appear in the vaulation file: {}".format(params[1]))
                
            cmd=99
            
####################################################################    


        elif cmd == 7: # HOW TO GET TO *
            
            """
            Key:
                S = Start Point
                X = Wall
                F = Floor
                G = Goal
            """            
            FLOOR = {
        
                "ecg":          ['SFFXFFXF',
                                 'XFFFFXFX',
                                 'XXFFXFFX',
                                 'XGXFXFXX',
                                 'XFFFFFFF'],
                          
                "exit":         ['SFFXFFXF',
                                 'XFXFFXFX',
                                 'XFFFXFFX',
                                 'XXXFXFXX',
                                 'XFXFFFFG'],
                          
                "reception":    ['SFFXFFGF',
                                 'XFXFFXFX',
                                 'XFFFXFFX',
                                 'XXXFXFXX',
                                 'XFXFFFFF'],
                           
                "toilet":       ['SFFXFFXF',
                                 'XFHFFXFX',
                                 'XFFFXFFX',
                                 'XXXFXFXG',
                                 'XFXFFFFF']
                    }
            
    
            
       
            WEIGHTS = {
                    "ecg":          "dqn_ecg_weights.h5f",
                    "exit":         "dqn_exit_weights.h5f",
                    "reception":    "dqn_reception_weights.h5f",
                    "toilet":       "dqn_toilet_weights.h5f"
                    }
                    
 
            params[1] = params[1].lower()
        
            while True:
                if params[1] in FLOOR:
                    
                    desc = FLOOR[params[1]]
                    desc = np.asarray(desc,dtype='c')
              
                    env = gym.make('FrozenLake-v0',
                                   desc=desc,
                                   is_slippery=False)
                    env.render()
                    
                    countActions = env.action_space.n
                    countstates = env.observation_space.n
                    
                    # run DQN model with saved weights
                    rlModelFloorRoute(countActions,
                                      countstates,
                                      env,
                                      WEIGHTS[params[1]])
                    break
                     
                else:
                    print("Error! invlaid floor plan. Please select one of the following: {}".format("ecg, exit, reception, or toilet"))
                    break
                    #raise Exception ("error! invlaid floor plan. Please select one of the following: {}".format("ecg, exit, reception, or toilet"))
            cmd=99
            
 ####################################################################  
            
            
        elif cmd == 8: # CALL 999
            
            """
            Key:
                Purple = pickup location
                Blue = destination 
                Red ambulance = empty 
                Yellow ambulance = pick up action
                Green ambulance = full ambulance

            """
             
            ENV_NAME = "Taxi-v3"
            env = gym.make(ENV_NAME)
            
            #randomise start point
            env.seed(np.random.seed(123))
            
            countActions = env.action_space.n
            countstates = env.observation_space.n
            
            # run DQN model with saved weights
            rlModelEmergencyRoute(countActions,countstates, env)
            
            
            cmd=99
            
####################################################################   
            
            
        elif cmd == 9: # EN TO FR
         
           
            model = load_model("new_eng-fr_model.h5")
            
            # loads variables: purpose of this is to speed up preprocessing time by having it read into from a file
            # by default, this would take ~15secs; now has been shorten to almost instant
            with open('loadPreprocess.pkl', 'rb') as f:
               preprocessEnData, preprocessFrData, enTokenize, frTokenize = pickle.load(f)
    
            userSentence = input("Enter text:  ").lower()
          
    
            sentence = [enTokenize.word_index.get(word) for word in userSentence.split()]
            sentence = pad_sequences([sentence], maxlen=21, padding='post')
            print("French Translation: ", logits_to_text(model.predict(sentence)[0], frTokenize))
            
            cmd=99
            

            
          
#################################################################### 
            
            
        elif userInput != None:
            pprint(response(userInput))
            sent_tokens.remove(userInput)
            
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)
