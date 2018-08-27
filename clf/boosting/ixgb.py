from clf.base import BaseModel
from xgboost import XGBClassifier


class Model(BaseModel):

    params = {"objective": "multiclass:softmax",
              "reg_lambda": 0.8,
              "reg_alpha": 0.4,
              "max_depth": 10,
              "max_delta_step": 1}

    model = XGBClassifier(**params)

    def __init__(self):
        super().__init__(Model.model)
