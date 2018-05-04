from sklearn.covariance import EllipticEnvelope


class ICovariance():

    CONTAMINAZTION = 0.012396153326979478

    def __init__():
        pass

    def fit(self,x_train):
        clf = EllipticEnvelope(contamination=ICovariance.CONTAMINAZTION)
        clf.fit(x_train)
        # clf.score(knownone,knownonelabel)
        