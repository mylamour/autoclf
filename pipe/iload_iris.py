from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd

def itrain_pipe():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2, random_state=42 )
    
    return x_train, y_train, x_test, y_test


def ipredict_pipe():
    iris = load_iris()
    return iris.data    
