from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.decomposition import PCA, NMF
from sklearn.svm import SVC



class Model():
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

    def __init__(self, l=0.01):
        self.l = l
        self.model = None

    def fit(self,x_train, y_train):
        self.model = GridSearchCV(Model.pipe, cv=3, n_jobs=-1, param_grid=Model.param_grid)
        self.model = self.model.fit(x_train,y_train)

    def score(self,x_test,y_test):
        return self.model.score(x_test,y_test)
    
    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_params(self, deep = False):
        """
            for cross_validation:
                https://stackoverflow.com/questions/20330445/how-to-write-a-custom-estimator-in-sklearn-and-use-cross-validation-on-it
        """
        return {'l':self.l}