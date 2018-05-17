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
└── models.py│
├── train.py
└── predict.py
```

* 一般情况 data 目录下放置数据集
* clf 文件夹下是为了自定义的机器学习算法，例如GridSearch SVC等, 而其子文件夹nn用于存放神经网络等深度学习算法
* pipe 文件夹下放置对数据集的预定义处理, 意味着你可以从任何地方加载并处理你的数据, 例如pipe/iload_aliatec.py即是对此次ATEC风险支付的数据处理
* saved 为了存放训练好的模型，或者预测后的数据

# Note

* 在load数据进行Pipline处理后，再交由自定义算法Pipline处理时可能会有意想不到的错误。(Sklearn本身的问题)，可以只在其中一处做Pipline,即只在pipe文件夹下load数据时自定义，也可以只在自定义算法时进行pipline

# Todo

- [ ] 增加requerments.txt 文件
- [ ] 单元测试
- [x] 伪ETL工程目录
- [ ] 性能评价模块
- [x] 动态创建类的函数
- [x] 自定义 nn 函数
- [x] 自定义 clf 函数
- [x] 支持自定义函数的`cross_validation`
- [x] 捕获ctrl+c，中断当前训练器
