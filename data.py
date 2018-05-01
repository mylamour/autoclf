import pandas as pd
from sklearn.model_selection import train_test_split

train_data_path = './atec_anti_fraud_train.csv'
predict_data_path = './atec_anti_fraud_test_a.csv'

DROPCOLUMS = ["id","label","date"]

# 0 .... 1, 0 is safe / 1 is not safe

def load_train_test():
    data = pd.read_csv(train_data_path)
    data = data.fillna(0)
    unlabeled = data[data['label'] == -1]
    labeled = data[data['label'] != -1]

    train, test = train_test_split(labeled, test_size=0.2, random_state=42)

    cols = [c for c in DROPCOLUMS if c in train.columns]
    x_train = train.drop(cols,axis=1)

    cols = [c for c in DROPCOLUMS if c in test.columns]
    x_test = test.drop(cols,axis=1)

    y_train = train['label']
    y_test = test['label']

    return x_train, y_train, x_test, y_test

def load_predict_data():
    upload_test = pd.read_csv(predict_data_path)
    upload_test = upload_test.fillna(0)
    upload_id = upload_test['id']
    
    cols = [c for c in DROPCOLUMS if c in upload_test.columns]
    upload_test = upload_test.drop(cols,axis=1)

    return upload_id, upload_test

def _load_iris_test():
    from sklearn.datasets import load_iris
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2, random_state=42 )

    return x_train, y_train, x_test, y_test