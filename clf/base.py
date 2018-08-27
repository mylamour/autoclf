
class BaseModel():
    """
        custom classifier
    """
    def __init__(self, model, classes=None, l=0.01,):
        self.l = l
        self.model = model
        self.classes = classes

    def fit(self,x_train, y_train):
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