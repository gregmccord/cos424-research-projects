# Overview

Our task for this project was to predict several key features for children growing up in split households and are in danger of growing up in poverty. This project was part of the [Fragile Families Challenge](https://ffcws.princeton.edu/news/fragile-families-challenge-mass-collaboration-paper-published), where we used the data collected by these researchers over several decades to try and predict several key attributes about children growing up during this circumstances that are thought to have a strong influence on their livelihood. Some of these response variables include whether or not their family was evicted before they were 15 or if they received some form of job training. Due to the sensitive nature of the project, I am unable to share the data associated with it, but I am able to share the code I wrote and discuss the results of the research. Please see the full paper under `/writeup`.

# Reproducibility

1. Run `python missingData.py` to generate `output.csv` from `background.csv` to fill missing data.
2. Run `python genData.py [all/both/question/constructed]` to create labeled versions of the training data matching the input specifications.
3. Run `python classificationDiscrete.py [eviction/layoff/jobTraining] [lr/mnb/gnb/svm/rf] [all/both/question/constructed]` generate model files:
    1. First parameter controls which discrete target you are predicting
    2. Second parameter controls the algorithm used for prediction
    3. Third parameter controls which dataset is used
4. Run `python predict.py [lr/mnb/gnb/svm/rf] [all/both/question/constructed]` to generate predictions for the specified model.
5. Run `python topTerms.py [all/both/question/constructed]` to find the top variables according to the random forest classifier.