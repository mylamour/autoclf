# Folder

* data 目录下放置数据集
* pipe 文件夹下放置对数据集的预定义处理。
* clf 文件夹下是为了自定义的机器学习算法，例如GridSearch SVC等


# Bug

* 在load数据进行Pipline处理后，再交由自定义算法Pipline处理时可能会有意想不到的错误。(Sklearn本身的问题)，可以只在其中一处做Pipline

# Todo

- [ ] 增加requerments.txt 文件
- [ ] 单元测试
- [x] 伪ETL工程目录
- [ ] 性能评价模块
- [x] 动态创建类的函数
- [x] 自定义 nn 函数
- [x] 自定义 clf 函数
- [ ] 捕获ctrl+c，中断当前训练器
