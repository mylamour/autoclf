每次重写data.py里的load_train_test，不如放置到单独的文件里用。
每次加载插件

clf 下面是为了自定义的常用机器学习算法，例如增加了GridSearch等

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


# Todo

- [ ] 增加requerments.txt 文件
- [ ] ETL 工程
- [ ] AUC、ROC 性能评价模块
- [x] 动态创建类的函数