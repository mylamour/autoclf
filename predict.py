import pandas as pd
import glob
import csv
from sklearn.externals import joblib

models = glob.glob('saved/*.pkl')

TESTFALG = False

if TESTFALG:
    from pipe import iload_iris_pipe
    x_train, y_train, x_test, y_test = iload_iris_pipe()

    for model in models:
        
        clf = joblib.load(model)
        modelname = clf.__class__.__name__
        if hasattr(clf, "predict"):
            print(clf.predict(x_test))
        if hasattr(clf,'predict_proba'):
            print(clf.predict_proba(x_test))

def common_save(predict):
    p = pd.DataFrame(predict,columns=["score"])
    res = pd.concat([data_id,p],axis=1)
    res.to_csv(filename,index=False)
    print("[+] Save Predict Result To {} Sucessful".format(filename))


def main():
    from pipe import iload_predict_data , isave_predict_data
    data_id, data_features = iload_predict_data()

    for model in models:
        
        clf = joblib.load(model)
        modelname = clf.__class__.__name__

        if hasattr(clf, "predict"):
            _ = clf.predict(data_features)
            save_predict = "saved/{}_predict.csv".format(modelname)
            isave_predict_data(data_id, _, save_predict)

        if hasattr(clf, 'predict_proba'):
            _ = clf.predict_proba(data_features)
            _ = [ 1-i[0] for i in _ ]
            save_predict_proba = "saved/{}_predict_proba.csv".format(modelname)
            isave_predict_data(data_id, _, save_predict_proba)

if __name__ == '__main__':
    main()

    # def write_predict(data_id, predicts):
    #     with open('{}.predict'.format(filename), 'w') as csvfile:
    #         fieldnames = ['id', 'score']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()

    #         for uuid, predict in zip(data_id,predicts):
    #             writer.writerow({'id': uuid, 'last_name': predict})
