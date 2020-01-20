0. BNB.ipynb is our Bernoulli Naive Bayes implementation

1. data_processing.py takes in the raw comment data and cleans it by: lowercasing all, removing punctuation, stemming, removing rare words, and tokenizing. 
It outputs train_clean.csv and test_clean.csv, which other scripts then take as inputs.

2. Optimization.ipynb handles the hyperparameter optimization process for all main models used

3. models.ipynb fits models with optimal parameters and outputs their predictions, which were used as Kaggle submissions

3. NN_ensemble.ipynb trains a neural network, but also includes the NN+SVM+MNB enseble that led us to our best Kaggle submission

4. model_trust_classifier.py implements our own ensemble classifier that optimizes over models weights
