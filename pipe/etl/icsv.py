import pandas as pd
import sys

def iload_csv(csvfile, label, header=None, dropcolumns=None):
    """
    process csv file with return training data
    """
    try:
        data = pd.read_csv(csvfile,header=None)
        data = data.fillna(0)

    except Exception as e:
        print("[X] FILE Error: File is not exist or not is csv format")
        sys.exit(1)

    if header:
        data.columns = header

    if  label in data.columns:
        target = data[label]
    else:
        print("[X] Params Error: Label Is Not Exists In DataFrame")
        sys.exit(1)

    if dropcolumns and type(dropcolumns) is list:
        cols = [c for c in dropcolumns if c in data.columns]
    else:
        cols = label
    
    data = data.drop(cols,axis=1)
    # target = data.loc[:, data.columns != label]

    return data, target

def groupby(dframe, column):
    """
        concatenate rows of pandas dataframe with same id
           A  B   id                A       B     id
        0  1  1   0                 
        1  2  1   0      -->        0  [1, 2]  [1, 1]   0
        2  3  2   1                 1  [3, 0]  [2, 2]   1
        3  0  2   1

    Get value with: data[data.columns[0]].values
    """
    data = dframe.groupby(column).agg(lambda x: x.tolist())
    return data, data.columns

def sortby(dframe, column):
    """
    apinfo.sort(key=lambda x: x[3])
    """
    pass

