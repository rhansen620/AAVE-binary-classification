#Binary Categorization: Create a machine to detect between "Standard" English and AAVE
#Imports

from typing import Iterator, Iterable, Tuple, Text, Union

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


NDArray = Union[np.ndarray, spmatrix]

#Open data file, create tuples of (label,text) and lists of texts/labels

detectlanguage = open("aave_train.txt",encoding="utf16") 
bm = [] #the tuples
texts = [] #list of text
labels = [] #list of labels
for line in detectlanguage:
    eachline = line.split("\t", 1) #create list with label, text
    label = eachline[0] #pull the label
    text = eachline[1] #pull the text

    stripline = text.strip() #remove white space if nccessary

    r =([label, stripline.rstrip("\n")]) #create tuple and remove new line markers

    bm.append(r)
    labels.append(label)
    texts.append(stripline)
    
#convert text to features

vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=10, max_df=.8)    
vectorizer.fit_transform(texts)

#Turn those features into a matrix!!
features = vectorizer.transform(texts).toarray()
print (features.shape)

#texts --> labels init
le = preprocessing.LabelEncoder()
le.fit(labels)

What are the labels?
list(le.classes_)

#Creates a label vector
label2 = le.transform(labels)

#Start the Logistic Regression
logreg = LogisticRegression(fit_intercept= True,class_weight= {0:3,1:9}, random_state = 0)

#Train
logreg.fit(features, label2, sample_weight=None)

#Evaluating the Training
train_acc = logreg.score(features, label2)
print("The Accuracy for Training Set is {}".format(train_acc*100))

#Preprocess the data for the prediction (test)
detectlanguage = open("aave_test.txt",encoding="utf16") 
texts1 = []
labelsone = []
for line in detectlanguage:
    eachline = line.split("\t", 1)
    label = eachline[0]
    text = eachline[1]
    stripline = text.strip()
    funtext = stripline.rstrip("\n")
    texts1.append(funtext)
    labelsone.append(label)
print (labelsone)

#create label vector for these
labels1 = le.transform(labelsone)
features1 = vectorizer.transform(texts1).toarray()
print (features1.shape)

#predict
predictions = logreg.predict(features1)

