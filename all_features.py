import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from sklearn.metrics import accuracy_score
import random
import time



features = pd.read_csv("features.csv", sep = ',')
#EEG Signal Data
eeg_X = features.drop('y',axis=1)

#EEG Signal Label
eeg_Y = features['y']

# Normalization of Features
#standardization
#standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(eeg_X)
scaled_features = scaler.transform(eeg_X)
eeg_features = pd.DataFrame(scaled_features)

#Cnvert Dataframe into Numpy Array for PSO
X = np.array(eeg_features)
y = np.array(eeg_Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(eeg_features,eeg_Y,test_size=0.20, random_state = 42)


t1 = time.time()
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

True_Positive = cm[0][0]
True_Negative = cm[1][1]
False_Positive = cm[0][1]
False_Negative = cm[1][0]

Accuracy = (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative) * 100
print("%.2f" % Accuracy)
Sensitivity = True_Positive / (True_Positive + False_Negative) * 100
print("%.2f" % Sensitivity)
Specificity = True_Negative / (True_Negative + False_Positive) * 100
print("%.2f" % Specificity)

print("Time required for Naive Bayes : {}".format(time.time()-t1))

t2 = time.time()
#SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
True_Positive = cm[0][0]
True_Negative = cm[1][1]
False_Positive = cm[0][1]
False_Negative = cm[1][0]
Accuracy = (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative) * 100
print("%.2f" % Accuracy)
Sensitivity = True_Positive / (True_Positive + False_Negative) * 100
print("%.2f" % Sensitivity)
Specificity = True_Negative / (True_Negative + False_Positive) * 100
print("%.2f" % Specificity)

print("Time required for SVM : {}".format(time.time()-t2))


t3 = time.time()
#KNN
from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
True_Positive = cm[0][0]
True_Negative = cm[1][1]
False_Positive = cm[0][1]
False_Negative = cm[1][0]
Accuracy = (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative) * 100
print("%.2f" % Accuracy)
Sensitivity = True_Positive / (True_Positive + False_Negative) * 100
print("%.2f" % Sensitivity)
Specificity = True_Negative / (True_Negative + False_Positive) * 100
print("%.2f" % Specificity)

print("Time required for KNN : {}".format(time.time()-t3))
