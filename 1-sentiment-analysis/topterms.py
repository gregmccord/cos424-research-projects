# Author: Gregory McCord
# NetID: gmccord

import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = np.genfromtxt(myfile,delimiter=',')
  return bagofwords

def read_vocab_file(myfile):
    with open(myfile) as f:
        return [line.rstrip('\n') for line in f]

def main(argv):
    path = 'data'

    # Ensure the input is correct
    if (len(argv) != 1):
        print('Usage: \n python classification.py [True/False]')
        sys.exit(2)

    # Will there be feature selection?
    if any(argv[0] == b for b in ["True","False"]):
        feature_selection = argv[0] == "True"
    else:
        print('Usage: \n python classification.py [True/False]')
        sys.exit(2)

    if feature_selection:
        bow = read_bagofwords_dat(path + '/out_bag_of_words_transform.csv')
        vocab = read_vocab_file(path + "/out_vocab_transform.txt")
    else:
        bow = read_bagofwords_dat(path + '/out_bag_of_words_5.csv')
        vocab = read_vocab_file(path + "/out_vocab_5.txt")
    classes = read_bagofwords_dat(path + '/out_classes_5.txt')

    # Perform classification using Random Forest using the gini criterion
    # without tuning the hyperparameter
    clf = RandomForestClassifier(random_state=0, n_estimators=500).fit(bow, classes)

    # get importances in sorted order
    importances = clf.feature_importances_
    index = range(0,len(importances))
    joint = list(zip(importances, index))
    joint.sort(key = lambda t: t[0], reverse=True)

    # Print top 20 vocab terms
    for x in range(20):
        b = joint[x][1]
        print(vocab[b])

if __name__ == "__main__":
    main(sys.argv[1:])