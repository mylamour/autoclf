import pandas as pd
import os

from sklearn.model_selection import train_test_split

# train_data_path = 'data/atec_anti_fraud_train.csv'
# predict_data_path = 'data/atec_anti_fraud_test_a.csv'

DROPCOLUMS = ["id","label","date"]
# 0 .... 1, 0 is safe / 1 is not safe

def iload_pipe(train_data_path):

    if os.path.isfile(train_data_path):
        print("[√] Path Checked, File Exists")
    else: 
        print("[X] Please Make Sure Your Datasets Was Exists")
        import sys
        sys.exit(1)
        
    data = pd.read_csv(train_data_path)
    data = data.fillna(0)
    labeled = data[data['label'] != -1]

    train, test = train_test_split(labeled, test_size=0.2, random_state=42)

    cols = [c for c in DROPCOLUMS if c in train.columns]
    x_train = train.drop(cols,axis=1)

    cols = [c for c in DROPCOLUMS if c in test.columns]
    x_test = test.drop(cols,axis=1)

    y_train = train['label']
    y_test = test['label']
    return x_train, y_train, x_test, y_test

def isave_data(predict_data_path):
    upload_test = pd.read_csv(predict_data_path)
    upload_test = upload_test.fillna(0)
    upload_id = upload_test['id']
    
    cols = [c for c in DROPCOLUMS if c in upload_test.columns]
    upload_test = upload_test.drop(cols,axis=1)

    return upload_id, upload_test




# def iload_aliatec_all(train_data_path,predict_data_path):
    
#     if os.path.isfile(train_data_path) and os.path.isfile(predict_data_path):
#         print("[√] Path Checked, File Exists")
#     else: 
#         print("[X] Please Make Sure Your Datasets Was Exists")
#         import sys
#         sys.exit(1)
        
#     data = pd.read_csv(train_data_path)
#     data = data.fillna(0)
#     unlabeled = data[data['label'] == -1]
#     labeled = data[data['label'] != -1]

#     knownzero = data[data['label']==0]
#     knownone = data[data['label']==1]

#     contamination = knownone.shape[0]/knownzero.shape[0] # 0.012396153326979478

#     knownzero_label = knownzero['label'] 
#     knownone_label = knownone['label'] 

#     X_train = pd.concat([knownone,knownzero],axis=0)
#     Y_train = pd.concat([knownone_label,knownzero_label],axis=0)

#     cols = [c for c in DROPCOLUMS if c in knownzero.columns]
#     knownzero = knownzero.drop(cols,axis=1)

#     cols = [c for c in DROPCOLUMS if c in knownone.columns]
#     knownone = knownone.drop(cols,axis=1)

#     cols = [c for c in DROPCOLUMS if c in unlabeled.columns]
#     unlabeled = unlabeled.drop(cols,axis=1)

#     import numpy as np
#     from sklearn.covariance import EllipticEnvelope
#     from sklearn.neighbors import LocalOutlierFactor

#     clf = LocalOutlierFactor(contamination=contamination)
#     clf.fit(X_train)

#     s1 = clf.score(knownone,[1 for i in range(knownone.shape[0])])
#     s2 = clf.score(knownone,[-1 for i in range(knownone.shape[0])])

#     predicts = clf.predict(unlabeled)
    
#     if s1 > s2:
#         s1_1 = clf.score(knownzero,[-1 for i in range(knownzero.shape[0])])
#         print("Inlier Test Score Vs Outlier Test Score: {},{}".format(s1,s1_1))
#         p_type = 1  # 1 is 1
        
#         predictone = np.where(predicts == 1)
#         predictzero = np.where(predicts == -1)
    

#     else:
#         s2_2 = clf.score(knownzero,[1 for i in rane(knownzero.shape[0])])
#         print("Inlier Test Score Vs Outlier Test Score: {},{}".format(s2,s2_2))
#         p_type = -1  # -1 is 1
       
#         predictone = np.where(predicts == -1)
#         predictzero = np.where(predicts == 1)

#     lof_one = unlabeled.iloc[[predictone]]
#     lof_zero = unlabeled.iloc[[predictzero]]
    
#     lof_train = pd.concat([lof_zero,lof_one],axis = 0)
#     lof_lables = [0 for i in range(lof_zero.shape()[0])] + [1 for i in range(lof_one.shape()[0])]
#     lof_lables = pd.DataFrame(lof_lables,columns=["label"])
    

#     X_train = pd.concat([X_train,lof_train],axis=0)
#     Y_train = pd.concat([Y_train,lof_lables], axis=0)

#     labeled = pd.concat([X_train,Y_train],axis=1)


#     train, test = train_test_split(labeled, test_size=0.2, random_state=42)
    
#     cols = [c for c in DROPCOLUMS if c in train.columns]
#     x_train = train.drop(cols,axis=1)

#     cols = [c for c in DROPCOLUMS if c in test.columns]
#     x_test = test.drop(cols,axis=1)
    
#     y_train = train['label']
#     y_test = test['label']

            
            


#     return x_train, y_train, x_test, y_test

