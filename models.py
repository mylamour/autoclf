import os

import pandas as pd
import numpy as np

# Sklearn Common Import
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import metrics

# Decomposition
# PCA 无监督， LDA 有监督
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Some Classifier Algorithms

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
    ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def dumpit(clf, modeldumpname):
    if os.path.isfile(modeldumpname):
        print("[!] {} Existed".format(modeldumpname))
        modeldumpname = "{}.second".format(modeldumpname)

    joblib.dump(clf, modeldumpname)
    print("[+] Save it in {}".format(modeldumpname))


def load_classifications(boosting=True, gridsearch=False):
    """
        please define your classifications here,
        must be key and value
    """

    if boosting:
        rf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
        adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                    algorithm="SAMME", n_estimators=200)
        xgb = XGBClassifier()
        lgb = LGBMClassifier()
    
    if gridsearch:
        """
            find igrid file to have a grid search
        """
        pass

    dt = DecisionTreeClassifier()  # max_depth=4a
    knn = KNeighborsClassifier()
    rbfsvc = SVC(kernel='rbf', probability=True)
    lg = LogisticRegression(random_state=1)
    gnb = GaussianNB()

    return locals()

class Model:

    def __init__(self, clfs,loss):
        self.clfs = clfs
        self._score = None
        self._clf = None        
        self.models = []
        self.loss = loss

    # def model(self,clf):
    #     return self.clf

    def fit(self,x_train, y_train=None, x_test=None,y_test=None, cv=None):
        try:
            iterator = iter(self.clfs)
        except TypeError:
            self._clf = self.clfs
            self.train(x_train, y_train,x_test, y_test)
            self.models.append(self._clf)
            
        else:
            for clf in self.clfs:
                self._clf = clf
                self.train(x_train, y_train,x_test,y_test)
                self.models.append(self._clf)

    def train(self, x_train, y_train=None, x_test=None,y_test=None):
        name = self._clf.__class__.__name__
        print("[*] Now Training With {:<10s}".format(name), end="")
        self._clf.fit(x_train, y_train)
        cross_validate(self._clf,x_train,y_train)
        print(" {} : {} \t".format(self.loss,self.evaluation(y_test, self._clf.predict_proba(x_test))))
        print("Report:{}".format(metrics.classification_report(y_test,self._clf.predict(x_test))))
        self.score(x_test,y_test)
        print(" And Model Scores {:<10}".format(self._score))
        
        return self._clf
    
    def score(self, x_test, y_test=None):
        name = self._clf.__class__.__name__
        self._score = self._clf.score(x_test,y_test)
        return self._score

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def evaluation(self, y_test, y_pred):

        if self.loss == "neg_log_loss" or "log_loss":
            return metrics.log_loss(y_test,y_pred)

        if self.loss == "brier_score_loss" or "bs":
            return metrics.brier_score_loss(y_test, y_pred)

        if self.loss == "ranking_loss":
            return metrics.label_ranking_loss(y_test,y_pred)
        

    def save(self):
        for model in self.models:
            name = model.__class__.__name__
            import time
            modeldumpname = "saved/{}-{}.pkl".format(name.lower(),time.ctime().replace(" ","-"))
            dumpit(model, modeldumpname)