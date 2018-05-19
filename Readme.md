# Intro
一个`Mini`型的ml/dl的项目,需要使用者具有一定的编程能力。目录结构为

```
├── clf
│   │
│   ├── nn
│
├── data
│
├── pipe
│
└── saved
│
└── models.py
├── train.py
└── predict.py
```

* 一般情况 data 目录下放置数据集
* `clf` 文件夹下是为了自定义的机器学习算法，例如`GridSearch SVC`等, 而其子文件夹nn用于存放神经网络等深度学习算法
* `pipe` 文件夹下放置对数据集的预定义处理, 意味着你可以从任何地方加载并处理你的数据, 例如`pipe/iload_aliatec.py`即是对此次ATEC风险支付的数据处理
* `saved` 为了存放训练好的模型，或者预测后的数据


# Useage:

## train.py
```
$ python train.py

Usage: train.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  classification  this for select classification model
  cluster         this for select cluster model
```

```
$ python train.py classification --help

Usage: train.py classification [OPTIONS]

  this for select classification model

Options:
  --method TEXT               Your method for training model
  --pipe TEXT                 Data Pipe Line File
  --cross-validation INTEGER  Cross Validation
  --help                      Show this message and exit.

```

```
$ python train.py classification --pipe pipe/iload_digits.py --method lg --method rbfsvc

[*] Now Training With LogisticRegression And Model Scores 0.9666666666666667
[*] Now Training With SVC        And Model Scores 0.9805555555555555
[+] Save it in saved/logisticregression.pkl
[+] Save it in saved/svc.pkl

```

```
$ python train.py classification --pipe pipe/iload_digits.py --method lg

[*] Now Training With LogisticRegression And Model Scores 0.9666666666666667
[!] saved/logisticregression.pkl Existed
[+] Save it in saved/logisticregression.pkl.second

```

```
$ python train.py classification --pipe pipe/iload_iris.py

[!] Now We Will Use Default All Method
[*] Now Training With VotingClassifier And Model Scores 1.0
[*] Now Training With VotingClassifier And Model Scores 1.0
[*] Now Training With AdaBoostClassifier And Model Scores 0.9333333333333333
[*] Now Training With GaussianNB And Model Scores 1.0
[*] Now Training With XGBClassifier And Model Scores 1.0
[*] Now Training With LogisticRegression And Model Scores 1.0
[*] Now Training With SVC        And Model Scores 1.0
[*] Now Training With KNeighborsClassifier And Model Scores 0.9666666666666667
[*] Now Training With RandomForestClassifier And Model Scores 1.0
[*] Now Training With DecisionTreeClassifier And Model Scores 1.0
[*] Now Training With IGridSVC   And Model Scores 1.0
[!] saved/votingclassifier.pkl Existed
[+] Save it in saved/votingclassifier.pkl.second
[!] saved/votingclassifier.pkl Existed
[+] Save it in saved/votingclassifier.pkl.second
[!] saved/adaboostclassifier.pkl Existed
[+] Save it in saved/adaboostclassifier.pkl.second
[!] saved/gaussiannb.pkl Existed
[+] Save it in saved/gaussiannb.pkl.second
[!] saved/xgbclassifier.pkl Existed
[+] Save it in saved/xgbclassifier.pkl.second
[!] saved/logisticregression.pkl Existed
[+] Save it in saved/logisticregression.pkl.second
[!] saved/svc.pkl Existed
[+] Save it in saved/svc.pkl.second
[!] saved/kneighborsclassifier.pkl Existed
[+] Save it in saved/kneighborsclassifier.pkl.second
[!] saved/randomforestclassifier.pkl Existed
[+] Save it in saved/randomforestclassifier.pkl.second
[!] saved/decisiontreeclassifier.pkl Existed
[+] Save it in saved/decisiontreeclassifier.pkl.second
[!] saved/igridsvc.pkl Existed
[+] Save it in saved/igridsvc.pkl.second
```

## predict.py
载入saved文件夹下的已经保存好的模型进行预测，默认预测结果输出到`saved`文件夹，分别以`predict`和`proba`的后缀结尾，还可以通过，自定义输出路径，指定预测结果的输出，例如`--out woqu`

```
$ python predict.py predict  --help
Usage: predict.py predict [OPTIONS]

Options:
  --method TEXT  Your method for training model
  --pipe TEXT    Data Pipe Line File
  --out TEXT     Directory for save predict
  --help         Show this message and exit.

```

```
$ python predict.py predict  --pipe pipe/iload_iris.py --method saved/adaboostclassifier.pkl

 [####################################]  100% predict use model: AdaBoostClassifier

```

```
$python predict.py predict  --pipe pipe/iload_iris.py --method saved/adaboostclassifier.pkl --method saved/decisiontreeclassifier.pkl
  
  [##################------------------]   50% predict use model: AdaBoostClassifier
  [####################################]  100% predict use model: DecisionTreeClassifier

```

```
$ python predict.py predict  --pipe pipe/iload_iris.py --method saved
Use Batch Models From /home/mour/MlDl/autoclf/saved
  [###---------------------------------]   10% predict use model: LogisticRegression
  [#######-----------------------------]   20% predict use model: AdaBoostClassifier
  [##########--------------------------]   30% predict use model: XGBClassifier
  [##############----------------------]   40% predict use model: SVC
  [##################------------------]   50% predict use model: GaussianNB
  [#####################---------------]   60% predict use model: KNeighborsClassifier
  [#########################-----------]   70% predict use model: VotingClassifier
  [############################--------]   80% predict use model: DecisionTreeClassifier
  [################################----]   90% predict use model: RandomForestClassifier
  [####################################]  100% predict use model: IGridSVC
```

```
$ python predict.py predict  --pipe pipe/iload_iris.py --method saved --out woqu
Use Batch Models From /home/mour/MlDl/autoclf/saved
  [###---------------------------------]   10% predict use model: LogisticRegression
  [#######-----------------------------]   20% predict use model: AdaBoostClassifier
  [##########--------------------------]   30% predict use model: XGBClassifier
  [##############----------------------]   40% predict use model: SVC
  [##################------------------]   50% predict use model: GaussianNB
  [#####################---------------]   60% predict use model: KNeighborsClassifier
  [#########################-----------]   70% predict use model: VotingClassifier
  [############################--------]   80% predict use model: DecisionTreeClassifier
  [################################----]   90% predict use model: RandomForestClassifier
  [####################################]  100% predict use model: IGridSVC

```

# Note

* 在load数据进行Pipline处理后，再交由自定义算法Pipline处理时可能会有意想不到的错误。(Sklearn本身的问题)，可以只在其中一处做Pipline,即只在pipe文件夹下load数据时自定义，也可以只在自定义算法时进行pipline

* 数据预处理文件的定义需要遵循格式，即要处理内容定义在`iload_pipe`函数中,预测函数定义在`ipredict_pipe`中

# Todo

- [ ] 增加requerments.txt 文件
- [ ] 单元测试
- [ ] 增加`cluster`算法相关
- [x] 重构`predict`文件
- [x] 伪ETL工程目录
- [ ] 性能评价模块
- [x] 动态创建类的函数
- [x] 自定义 nn 函数
- [x] 自定义 clf 函数
- [x] 支持自定义函数的`cross_validation`
- [x] 捕获ctrl+c，中断当前训练器
