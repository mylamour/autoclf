import os
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from sklearn.externals import joblib

def loadmodelfeatures(model_path):
    """
        load features weight, and return columns and weights
    """
    model = joblib.load(model_path)
    a_weight = model.feature_importances_
    s_weight = list(a_weight[a_weight!=0.0])
    z_weight = list(a_weight[a_weight==0.0])

    zero_weights_columns =  [ "f{}".format(i+1) for i in np.where(a_weight==0.0)[0]]
    s_weight_columns = [ "f{}".format(i+1) for i in np.where(a_weight!=0.0)[0]]

    return s_weight, s_weight_columns, z_weight, zero_weights_columns

def multicall(p,l):
    """
        return res with mutliprocessed of list
        example:
            multi process read files of folder
    """
    res = None
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(p,l))
    return res

def walkfolder(folder):
    files = []
    for r,d,f in os.walk(folder):
        for item in f:
            files.append(os.path.join(r,item))
    return files