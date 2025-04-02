# Overview

Our task for this project was fairly open-ended - the goal was to analyze the Netflix challenge dataset and come up with our own goals. One direction we could go down was to use something like LDA or clustering to uncover latent structures present in the data. I was interested in scraping the genre data to see if it was possible to predict the ratings someone would give to a movie based on their ratings and genre information of movies they had alraedy watched. Gathering and cleaning the data ended up being much more complicated than I expected, but it ended up providing a very rich source of data to explore. I also took the chance to work with sparse matrices (given the nature of the dataset) and to implement an Expectation Maximization (EM) algorithm to tune the model parameters.

# Reproducibility

1. Run `python imdb_genre.py` in order to generate the output files. When I originally ran this, IMDb only supported querying a single movie at a time with a strict rate limit. I don't believe this has since changed, but as a result, my initial code runs verys slowly and requires manually setting changing which group of movies to query at once before merging. If you don't want to rerun the script, the full output of this file is saved in `data/movie_titles_genre.txt`.
2. Run `python data_cleaning.py`, which pulls data from the assembled `movie_titles_genre.txt` file and the provided `netflix_full_csr.npz`, which is a sparse representation of the ratings matrix. It will then output plots for picking thresholds as well as truncated movie and ratings lists called `movies_small.csv` and `netflix_small.npy`.
3. Run `python analysis.py`, which establishes a baseline and then runs an EM/CD algorithm to optimize the parameters.
