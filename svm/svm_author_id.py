#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# reducedSize = int(len(features_train) / 100)
# features_train = features_train[: reducedSize]
# labels_train = labels_train[: reducedSize]

#########################################################
### your code goes here ###

#########################################################
from sklearn import svm

clf = svm.SVC(C=10000,kernel='rbf')

t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0), "s")

t1 = time()
pred = clf.predict(features_test)
print("predicting time: ", round(time()-t1), "s")

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, pred), pred[10],pred[26],pred[50])

import numpy as np
unique_elements, counts_elements = np.unique(pred, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
