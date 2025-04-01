# Author: Gregory McCord
# NetID: gmccord

import sys

import pandas as pd

def main(argv):
    # Ensure the input is correct
    if (len(argv) != 1):
        print('Usage: \n python genData.py [all/both/question/constructed]')
        sys.exit(2)

    dataset = argv[0]

    df = pd.read_csv("data/output.csv", low_memory=False)
    labels = pd.read_csv("data/train.csv", low_memory=False)
    meta = pd.read_csv("data/metadata.csv", low_memory=False, encoding='ISO-8859-1')
    indices = range(len(meta))

    # Perform feature selection
    mask = meta[dataset].tolist()
    mask = [bool(b) for b in mask]
    indices = [i for i in indices if mask[i]]
    df = df.iloc[:,indices]

    # True categorical variables
    cats = ['d3a23_a','d3a33a_14o','d3c12_b1','d3c12_b2','d3c13','d3c14',
        'd3g22a_9ot', 'd3g23a_got','r3a10_a6ot','r3a16_b1','r3a16_b2',
        'r3a18','r3b26_a','r3b13_a9ot','r3b31a_a','r3b31a_b','r3b36_a',
        'r3f1','r3f2','r3f7_a','p4g20_ot']
    
    # Replace other in mixed data type columns
    cols = df.select_dtypes(['object']).columns
    if len(set(cats) & set(cols)) > 0:
        cols = cols.drop(cats)
        df[cols] = df[cols].apply(lambda x: x.replace('Other',1))

        # factorize the categorical data
        df[cats] = df[cats].apply(lambda x: x.astype('category').cat.codes)
    else:
        df[cols] = df[cols].apply(lambda x: x.replace('Other',1))

    # Output 2 datasets - one with the training labels and one without
    df.to_csv("data/%s_data_pre.csv" % (dataset), index=False)
    df = df.merge(labels)
    df.to_csv("data/%s_data.csv" % (dataset), index=False)

if __name__ == "__main__":
    main(sys.argv[1:])