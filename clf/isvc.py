from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.decomposition import PCA, NMF
from sklearn.svm import SVC

from clf.base import BaseModel

class Model(BaseModel):
    N_FEATURES_OPTIONS = [2, 4]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
            {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
            {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },]

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', SVC( kernel="linear", probability=True))
    ])

    model = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid)

    def __init__(self, l=0.01):
        super().__init__(Model.model,l)