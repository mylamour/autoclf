import os

import pandas as pd
import numpy as np

# Sklearn Common Import
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

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

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


from clf import IGridSVC


def dumpit(clf, modeldumpname):
    if os.path.isfile(modeldumpname):
        print("[!] {} Existed".format(modeldumpname))
        modeldumpname = "{}.second".format(modeldumpname)

    joblib.dump(clf, modeldumpname)
    print("[+] Save it in {}".format(modeldumpname))


def load_classifications():
    """
        please define your classifications here,
        must be key and value
    """
       
    igridsvc = IGridSVC()
    
    dt = DecisionTreeClassifier()  # max_depth=4a
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=7)
    rbfsvc = SVC(kernel='rbf', probability=True)
    lg = LogisticRegression(random_state=1)
    xgb = XGBClassifier()
    gnb = GaussianNB()
    adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                       algorithm="SAMME", n_estimators=200)

    Voting1 = VotingClassifier(
        estimators=[
            ('dt', dt),
            ('knn', knn),
            ('rbfsvc', rbfsvc),
            ('lg', lg),
            ('xgb', xgb)
        ],
        voting='soft'
    )

    Voting2 = VotingClassifier(
        estimators=[
                    ('dt', dt),
                    ('knn', knn),
                    ('svc', rbfsvc),
                    ('lg', lg)
        ],
        voting='soft'
    )

    return locals()

class Model:

    def __init__(self, clfs):
        self.clfs = clfs
        self._score = None
        self._clf = None        
        self.models = []

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

    def save(self):
        for model in self.models:
            name = model.__class__.__name__
            modeldumpname = "saved/{}.pkl".format(name.lower())
            dumpit(model, modeldumpname)
            
# def train(x_train,y_train,x_test,y_test):

    # for clf in clfs:
    #     Flag = True
    #     name = clf.__class__.__name__
    #     modeldumpname = "saved/{}.pkl".format(name.lower())

    #     print("[*] Now Training With {:<10s}".format(name))

    #     try:
    #         clf.fit(x_train,y_train)
    #         cross_validate(clf,x_train,y_train)
    #     except KeyboardInterrupt:
    #         Flag = False
    #         print("[-] Skip {}".format(name))

    #     if Flag:
    #         score = clf.score(x_test,y_test)
    #         dumpit(clf,modeldumpname)
    #         print("[+] Saving Model {:<10s} with accuracy: {}".format(modeldumpname,score))

        # if not name.startswith("i"):    # custom class not implement cross valdation
        #     score = np.mean(cross_val_score(clf, x_train, y_train, cv=10))
        # else:
        #     score = np.mean(clf.cross_val_score(x_train,y_train,cv=10))


# if __name__ == '__main__':

#     print('Loading Data....',end='',flush=True)
#     from pipe import iload_iris_pipe, iload_aliatec_pipe
#     x_train, y_train, x_test, y_test = iload_iris_pipe()
#     print('\tDone')
#     train(x_train, y_train, x_test, y_test)

    # clf = XGBClassifier()

    # clf.fit(x_train,y_train)
    # score = clf.score(x_test,y_test)
    # modeldumpname = clf.__class__.__name__
    # dumpit(clf,modeldumpname)
    # print("[+] Saving Model {:<10s} with accuracy: {}".format(modeldumpname,score))
