import pandas as pd
import glob
import csv
import logging
from sklearn.externals import joblib
from .data import load_predict_data

models = glob.glob('./*.pkl')

data_features = pd.read_csv('atec_anti_fraud_test_a.csv')
data_features = data_features.fillna(0)
data_id = data_features['id']
data_features = data_features.drop(['id','date'],axis=1)

data_id, data_features = load_predict_data()


# def write_predict(data_id, predicts):
#     with open('{}.predict'.format(filename), 'w') as csvfile:
#         fieldnames = ['id', 'score']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         for uuid, predict in zip(data_id,predicts):
#             writer.writerow({'id': uuid, 'last_name': predict})

def write_predict(data_id,predict,filename):
    p = pd.DataFrame(predict,columns="score")
    res = pd.concat([data_id,b],axis=1)
    res.to_csv(filename,index=False)
    print("[+] Save Predict Result To {} Sucessful".format(filename))


for model in models:
    
    clf = joblib.load(model)
    modelname = clf.__class__.__name__

    if hasattr(clf, "predict"):
        _ = clf.predict(data_features)
        save_predict = "{}.predict".format(modelname)
        write_predict(data_id, _, save_predict)

    # if hasattr(clf, "predict_proba"):
    #     _ = clf.predict_proba(data_features)
    #     save_predict_proba = "{}.predict_proba".format(modelname)
    #     # write_predict_p(data_id, _, save_predict_proba)

