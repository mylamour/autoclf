import os

import pandas as pd
import numpy as np

# Sklearn Common Import
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# Decomposition
# PCA 无监督， LDA 有监督
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Some Classifier Algorithms

from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier,\
                             ExtraTreesClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Keras Binding

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical


# Keras Model/Layers

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation, Conv2D, \
                         MaxPool2D, Dropout, Flatten, Embedding,Reshape,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from .data import load_train_test

x_train, y_train, x_test, y_test = load_train_test()


clf1 = DecisionTreeClassifier()     #max_depth=4
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = LogisticRegression(random_state=1)
clf5 = XGBClassifier()
clf6 = GaussianNB()
clf7 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",n_estimators=200)


voting1 = VotingClassifier(
    estimators=[
        ('dt',clf1),
        ('knn', clf2),
        ('svc', clf3),
        ('lg',clf4)    
    ],
    voting='soft'
)

voting2 = VotingClassifier(
    estimators=[
        ('dt',clf1),
        ('knn', clf2),
        ('svc', clf3),
        ('lg',clf4),    
        ('xgb',clf5)    
    ],
    voting='soft'
)

clfs = [
    clf1,clf2,clf3,clf4,clf5,clf6,clf7,voting1,voting2
]

for clf in clfs:
    name = clf.__class__.__name__
    modeldumpname = "{}.pkl".format(name.lower())

    print("[*] Now Training With {:<10s}".format(name))

    cross_val_score(clf,x_train,y_train)
    score = clf.score(x_test,y_test)

    if os.path.isfile(modeldumpname):
        print("[x] {} Already Exists".format(modeldumpname))
        modeldumpname = "{}.second".format(modeldumpname)
        print("[-] Rename {}".format(modeldumpname))
        
    joblib.dump(clf,modeldumpname)

    print("[+] Saving Model {:<10s} with accuracy: {}".format(modeldumpname,score))
