# Overview

Our task for this project was to perform sentiment analysis on a provided data set (aggregated from various review sites). The train and test splits were provided to us along with the `preprocessSentences.py` script, which I slightly edited for ease and efficiency of use. They also provided the `testStandfordNLP.py` script for comparison against an (at-the-time) state-of-the-art sentiment analysis classifier. I have also since ported this code to Python 3. I've included the steps to generate the data below, which will help recreate all of the results seen in the paper. Output files have also been preserved in `data/archived_output.zip`. For the full results and to learn more about the process, please take a look at the full paper under `writeup/`.

# Reproducibility

1. Setup your python environment using requirements.txt in the root directory. I am using Python 3.7.4 in a virtual environment.
2. Run `python preprocessSentences.py` to create the output embeddings for the train data set.
3. Run `python featureSelection.py` to perform feature selection on the embeddings using a LinearSVC.
   1. This program performs 5-fold cross-validation to tune the hyperparameters.
   2. Once tuned, it outputs the "\_transform" embeddings.
4. Generate the embeddings for the test dataset:
   1. Run `python preprocessSentences.py -v out_vocab_5.txt` to create the output embeddings for the regular test data set.
   2. Run `python preprocessSentences.py -v out_vocab_transform.txt` to create the output embeddings for the transformed test data set.
5. Run the classifier for any combination of regular/transformed input and any of the supported classifiers
   1. Use the command `python classification.py [True/False] [svm/nb/knn/lr/rf]`.
   2. The random forest classifier use a set seed for reproducibility.
6. To sort the words in the vocabulary by importance in the random forest model by Gini Criterion, you can run `python preprocessSentences.py [True/False]` to use either the original or transformed dictionary.
7. You can then compare these results to the StanfordCoreNLP results
   1. Download the standalone server [here](https://stanfordnlp.github.io/CoreNLP/download.html)
   2. Launch the server with this command `java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000`
   3. Use the server to score the test dataset with `python testStandfordNLP.py`
   4. Close the server.
