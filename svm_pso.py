import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from sklearn.metrics import accuracy_score
import random
import pyswarms as ps
import time

t = time.time()
features = pd.read_csv("features.csv", sep = ',')

#EEG Signal Data
eeg_X = features.drop('y',axis=1)

#EEG Signal Label
eeg_Y = features['y']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(eeg_X)
scaled_features = scaler.transform(eeg_X)
eeg_features = pd.DataFrame(scaled_features)

#Cnvert Dataframe into Numpy Array for PSO
X = np.array(eeg_features)
y = np.array(eeg_Y)

#SVM
from sklearn.svm import SVC
# Create an instance of Naive Bayes
classifier = SVC()

# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 30
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

# The PSO Process
def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = 30 # dimensions should be the number of features

#Optimize
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
#Note : We can change the number of iterations to optimize more
# Perform optimization
cost, pos = optimizer.optimize(f, iters=100)

selected_features = X[:,pos==1]  # subset
# Create dataframe of selected features
selected_features = pd.DataFrame(selected_features)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(selected_features,eeg_Y,test_size=0.20, random_state=42)

#SVM
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train,y_train)

predictions = svm.predict(X_test)

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

print("Time required for SVM PSO : {}".format(time.time()-t))
