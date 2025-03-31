# Author: Gregory McCord
# NetID: gmccord

import numpy as np
import math
import sys

from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = np.genfromtxt(myfile,delimiter=',')
  return bagofwords

# The sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perform classification using Support Vector Machine with l1 penalty and linear kernel
def svm(bow, classes, test, y):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxC = 0
    for c in np.arange(0.1,1,0.1):
        clf = LinearSVC(C=c, penalty="l1", dual=False).fit(bow, classes)
        scores = cross_val_score(clf, bow, classes, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (c, mean))
        if mean > maxScore:
            maxScore = mean
            maxC = c

    # Train the model
    print("MaxC: %f" % maxC)
    print("MaxScore: %f" % maxScore)
    clf = LinearSVC(C=maxC, penalty="l1", dual=False).fit(bow, classes)

    # Predict labels for the test data
    pred = clf.predict(test)
    confidence = clf.decision_function(test)
    
    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    pred = sigmoid(confidence)
    roc_score = roc_auc_score(y, pred)
    print("ROC score: %f" % roc_score)

# Perform classification using Multinomial Naive Bayes
def nb(bow, classes, test, y):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxA = 0
    for a in np.arange(0,1,0.1):
        clf = MultinomialNB(alpha=a).fit(bow, classes)
        scores = cross_val_score(clf, bow, classes, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (a, mean))
        if mean > maxScore:
            maxScore = mean
            maxA = a

    # Train the model
    print("MaxA: %f" % maxA)
    print("MaxScore: %f" % maxScore)
    clf = MultinomialNB(alpha=maxA).fit(bow, classes)

    # Predict labels for the test data
    pred = clf.predict(test)
    pred_prob = clf.predict_proba(test)
    
    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y, prob)
    print("ROC score: %f" % roc_score)

# Perform classification using K-Nearest Neighbors
def knn(bow, classes, test, y):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxK = 0
    for k in np.arange(1,16,1):
        clf = KNeighborsClassifier(n_neighbors=k).fit(bow, classes)
        scores = cross_val_score(clf, bow, classes, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (k, mean))
        if mean > maxScore:
            maxScore = mean
            maxK = k

    # Train the model
    print("MaxK: %f" % maxK)
    print("MaxScore: %f" % maxScore)
    clf = KNeighborsClassifier(n_neighbors=maxK).fit(bow, classes)

    # Predict labels for the test data
    pred = clf.predict(test)
    pred_prob = clf.predict_proba(test)

    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y, prob)
    print("ROC score: %f" % roc_score)

# Perform classification using Logistic Regression using the l2 penalty
def lr(bow, classes, test, y):

    # Tune the hyperparameter
    maxScore = float("-inf")
    maxC = 0
    for c in np.arange(0.1,1,0.1):
        clf = LogisticRegression(penalty="l2", C=c).fit(bow, classes)
        scores = cross_val_score(clf, bow, classes, cv=5)
        mean = np.mean(scores)
        print("C: %f and Score: %f" % (c, mean))
        if mean > maxScore:
            maxScore = mean
            maxC = c

    # Train the model
    print("MaxC: %f" % maxC)
    print("MaxScore: %f" % maxScore)
    clf = LogisticRegression(penalty="l2", C=maxC).fit(bow, classes)

    # Predict labels for the test data
    pred = clf.predict(test)
    pred_prob = clf.predict_proba(test)

    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y, prob)
    print("ROC score: %f" % roc_score)

# Perform classification using Random Forest using the gini criterion
# without tuning the hyperparameter
def rf(bow, classes, test, y):

    # Train the model
    clf = RandomForestClassifier(random_state=0, n_estimators=500).fit(bow, classes)

    # Predict labels for the test data
    pred = clf.predict(test)
    pred_prob = clf.predict_proba(test)

    # Calculate the misclassification rate
    mc_rate = zero_one_loss(y, pred)
    print("MC rate: %f" % mc_rate)

    # Calculate the ROC curve
    prob = pred_prob[:,1:]
    roc_score = roc_auc_score(y, prob)
    print("ROC score: %f" % roc_score)

def main(argv):
    path = 'data'

    # Ensure the input is correct
    if (len(argv) != 2):
        print('Usage: \n python classification.py [True/False] [svm/nb/knn/lr/rf]')
        sys.exit(2)

    # Will there be feature selection?
    if any(argv[0] == b for b in ["True","False"]):
        feature_selection = argv[0] == "True"
    else:
        print('Usage: \n python classification.py [True/False] [svm/nb/knn/lr/rf]')
        sys.exit(2)

    if feature_selection:
        bow = read_bagofwords_dat(path + '/out_bag_of_words_transform.csv')
        classes = read_bagofwords_dat(path + '/out_classes_5.txt')
        test = read_bagofwords_dat(path + '/test_transform_bag_of_words_0.csv')
        y = read_bagofwords_dat(path + '/test_transform_classes_0.txt')
    else:
        bow = read_bagofwords_dat(path + '/out_bag_of_words_5.csv')
        classes = read_bagofwords_dat(path + '/out_classes_5.txt')
        test = read_bagofwords_dat(path + '/test_bag_of_words_0.csv')
        y = read_bagofwords_dat(path + '/test_classes_0.txt')

    # Which classifier will be used?
    if any(argv[1] == b for b in ["svm","nb","knn","lr","rf"]):
        if (argv[1] == "svm"):
            svm(bow, classes, test, y)
        elif (argv[1] == "nb"):
            nb(bow, classes, test, y)
        elif (argv[1] == "knn"):
            knn(bow, classes, test, y)
        elif (argv[1] == "lr"):
            lr(bow, classes, test, y)
        elif (argv[1] == "rf"):
            rf(bow, classes, test, y)
    else:
        print('Usage: \n python classification.py [True/False] [svm/nb/knn/lr/rf]')
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])