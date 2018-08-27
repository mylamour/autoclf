# Code Snippetss


* 动态创建类的函数

```python
def set_class_func(CLASS, FUNC):
    try:
        funclist = [i for i in dir(FUNC) if callable(getattr(FUNC,i)) and not i.startswith('_') ]
        for func in funclist:
            setattr(CLASS,str(func),classmethod(func))    
    except Exception as e:
        import sys
        print("Error When Create Class Method From {} TO {}".format(CLASS,FUNC))
        sys.exit(1)
```

* 文件夹路径
```python
import os
print(os.path.dirname(os.path.abspath("__file__")))
```

* print without new line
```python
print('Loading Data...',end='')
print('Done)
```

* AttributeError: 'LinearSVC' object has no attribute 'predict_proba', https://stackoverflow.com/questions/35076586/linearsvc-vs-svckernel-linear-conflicting-arguments
```python

LinearSVC(probability=True) # 提示没有该属性
#Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in  the choice of penalties and loss functions and should scale better to large numbers of samples.
# http://scikit-learn.org/stable/modules/svm.html
SVC( kernel="linear", probability=True)

```

* 添加自定义函数的`cross_validation`支持, https://stackoverflow.com/questions/20330445/how-to-write-a-custom-estimator-in-sklearn-and-use-cross-validation-on-it
  
```
    class A:
        def __init__(self,l):
            self.l = l

        def get_params(self, deep = False):
            return {'l':self.l}

```

* 生成requirements.txt
```
pipreqs .
```
* `LinearSVC` 没有 `predict_proba` https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
```
 from sklearn.calibration import CalibratedClassifierCV
 svm = LinearSVC()
 clf = CalibratedClassifierCV(svm) 
 clf.fit(X_train, y_train)
 y_proba = clf.predict_proba(X_test)

```

* BaggingClassifier is Roughly Equivalent to RandomForestClassifier 

```
from sklearn.tree import DecisionTreeClassifier
BaggingClassifier((DecisionTreeClassifier, splitter="random",max_leaf_nodes=16),
                n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)
```