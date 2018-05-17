
import numpy as np
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