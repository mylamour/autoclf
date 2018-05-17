DROPCOLUMS = ["id","label","date"]


def iload_pipe(train_data_path, predict_data_path):
    """
        load xgb model and select features
    """
    
    from sklearn.externals import joblib
    import numpy as np

    model = joblib.load('./saved/.pkl/five.pkl')

    fimportant = model.feature_importances_
    useless = [ "f{}".format(i+1) for i in list(np.where(fimportant==0)[0])]

    data = pd.read_csv(train_data_path)
    data = data.fillna(0)
    labeled = data[data['label'] != -1]


    unlabeled = data[data['label'] == -1]

    for_predict = unlabeled.drop(DROPCOLUMS,axis=1)
    predict = model.predict(for_predict)

    unlabeled = unlabeled.drop(['label'],axis=1)
    unlabeled = unlabeled[labeled.columns]

    data = pd.concat([labeled,unlabeled],axis=0)
    data = data.drop(useless,axis=1)

    pdata = pd.read_csv(predict_data_path)
    pdata = pdata.drop(useless,axis=1)

    data.to_csv("train_atec.csv",index=False)
    pdata.to_csv("test_atec.csv",index=False)

    print("Process from pretrained model was Done...")


def isave_data(data_id,predict,filename):
    p = pd.DataFrame(predict,columns=["score"])
    res = pd.concat([data_id,p],axis=1)
    res.to_csv(filename,index=False)
    print("[+] Save Predict Result To {} Sucessful".format(filename))
