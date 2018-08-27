from clf.base import BaseModel
from lightgbm import LGBMClassifier

"""
lgb tunning:
    https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
"""

class Model(BaseModel):

    params = {}
    params['learning_rate'] = 0.01
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'mutliclass'
    params['metric'] = 'softmax'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10

    model = LGBMClassifier(**params)

    def __init__(self):
        super().__init__(Model.model)
