""" Note: This classifier has been reimplemented from the default
VotingClassifier in scikit-learn.ensemble, due to the fact that
this classifier only supports training new models, and does not allow
a voting classifier to be built from pre-trained (ie pickled) models."""
import numpy as np
from mlxtend.regressor import StackingRegressor
from sklearn import clone
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from mlxtend.classifier import EnsembleVoteClassifier

class ModelTrustRegression:
    def __init__(self, model, n_neighbors=20, weights='uniform', n_folds=5):
        self.template_model = model
        self.n_neighbors = n_neighbors
        self.weights=weights
        self.n_folds = n_folds
        self.fold_regressions=[]
        self.fold_models=[]
        self.bagger = None

    def fit(self, X, values):
        #hard prediction
        for train_index, validation_index in KFold(n_splits=self.n_folds).split(X):
            train_set = X[train_index]
            train_values = values[train_index]

            validation_set = X[validation_index]
            validation_values = values[validation_index]

            fold_model = clone(self.template_model)
            fold_model.fit(train_set, train_values) #retrains a brand new model for the fold

            fold_regressor = KNeighborsRegressor(weights=self.weights, n_neighbors=self.n_neighbors)
            fold_regressor.fit(validation_set, fold_model.predict(validation_set) == validation_values)
            self.fold_regressions.append(fold_regressor)
            self.fold_models.append(fold_model)

        self.bagger = EnsembleVoteClassifier(self.fold_models, voting="soft", refit=False)
        self.bagger.fit(X, values) #trivial fit

    def predict(self, X):
        return np.mean([fm.predict(X) for fm in self.fold_regressions], axis=0)

    def predict_proba(self, X):
        return self.bagger.predict_proba(X)

    def get_bagger(self):
        return self.bagger


class ModelTrustClassifier:
    def __init__(self, models,  n_neighbors=20, weights='uniform', n_folds=5):
        self.models = models
        self.model_trust_regressions = []
        self.n_neighbors = n_neighbors
        self.weights=weights
        self.n_folds=n_folds

    def fit(self, X, values):
        for model in self.models:
            mtr = ModelTrustRegression(model, n_neighbors=self.n_neighbors, weights=self.weights, n_folds=self.n_folds)
            mtr.fit(X, values)
            self.model_trust_regressions.append(mtr)

    def predict(self, X):
        model_trusts = [trust_model.predict(X) for trust_model in self.model_trust_regressions]
        baggers = [trust_model.get_bagger() for trust_model in self.model_trust_regressions]
        classes = baggers[0].classes_.tolist()
        class_log_prob = []

        for j, bagger in enumerate(baggers):
            if(hasattr(bagger, "predict_proba")): #can predict porbability
                log_prob_map = lambda x, trust: np.log(x * trust)


                if not class_log_prob:
                    for i, row in enumerate(bagger.predict_proba(X)):
                        class_log_prob.append(log_prob_map(row, model_trusts[j][i]))

                else:
                    for i, row in enumerate(bagger.predict_proba(X)):
                        class_log_prob[i] += log_prob_map(row, model_trusts[j][i])


            else: #cannot predict porbability, so assume prob of 1 for actual pick and 0 otherwise
                if not class_log_prob:
                    for i, prediction in enumerate(bagger.predict(X)):
                        vote_row = np.full(bagger.classes_.shape[0], np.log(1-model_trusts[j][i]))
                        vote_row[classes.index(prediction)] = np.log(model_trusts[j][i])
                        class_log_prob.append(vote_row)

                else:
                    for i, prediction in enumerate(bagger.predict(X)):
                        vote_row = np.full(bagger.classes_.shape[0], np.log(1-model_trusts[j][i]))
                        vote_row[classes.index(prediction)] = np.log(model_trusts[j][i])
                        class_log_prob[i] += vote_row


        predictions = []
        for row in class_log_prob: #selecting prediction value with most probability
            vote_tuples = zip(row, classes)
            prediction = max(vote_tuples)[1]
            predictions.append(prediction)

        return predictions

    def get_mtr(self):
<<<<<<< HEAD
        return self.model_trust_regressions
=======
        return self.model_trust_regressions
>>>>>>> 54ce0a6b112b6ab73672a32908eb7bfc7849f4c7
