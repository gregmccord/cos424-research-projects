# Overview

For the final project, we were given the open-ended assignment to perform any analysis on any dataset of our choosing. I had the idea while exploring the Netflix data to see if there were any interesting relationships in movie scripts (rather than just recommending movies based on viewing/rating history) and if that could be used to show an evolution of language and genres over time. I found a relevant dataset on Kaggle from Cornell and made my initial proposal. I broke my research into two categories - first to try and classify a movie as being older or newer based on its language and second to uncover latent patterns in the dialogue between movies of different genres. I was particularly interested to see what patterns would emerge from movies that were set in different time periods. After completing my research and making several interesting discoveries, I collected my results and put out a paper in addition to a poster session to share the interesting findings. Please see the writeup section for the full results. It's worth noting that with the advent of modern LLMs, a new approach could be employed here as opposed to the tokenized bag of words model that could incorporate patterns like sentence structure in ways that were unavailable several years ago. Language models are an exciting realm of research, and I believe that this would be a very interesting avenue of research to pursue in the future.


# Reproducibility

1. Setting up the environment (all data and libraries have been included in the repository):
	1. [Download Mallet](http://mallet.cs.umass.edu/download.php) for LDA analysis
	2. [Download dialogue corpus](https://www.kaggle.com/Cornell-University/movie-dialog-corpus) from Kaggle
2. Run `python data.py [lr|nb|rf|lda|confusion]` depending on which classifier you want to use or if you want to generate the confusion matrix. This file handles:
	1. Preprocessing and tokenizing the data
	2. Assigning labels for binary classification
	3. Executes 1 of 3 families of functions:
		1. `lr/nb/rf` classifiers for new vs. old film based on the language
		2. `confusion` matrix for language used in each genre
		3. `lda` analysis to investigate the latent language space in each genre over time
