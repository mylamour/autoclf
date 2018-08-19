from models import Model, load_classifications, dumpit
import importlib
import os,sys
import time
import click


CLASSFICATIONS = load_classifications()

CLUSTERS = None


@click.group()
def cli():
    pass

@cli.command()
@click.option('--method',default='allmethod',help="Your method for training model")
@click.option('--pipe',default='None',help="Data Pipe Line File ")
@click.option('--cross-validation',default=10, help="Cross Validation ")
def cluster(method,pipe,cross_validation):
    """
        this for select cluster model
    """
    click.echo(click.style("[X] Temporaly Not Implement", fg='red', bg='black'))

@cli.command()
@click.option('--method',default='allmethod',help="Your method for training model")
@click.option('--pipe',default=None,help="Data Pipe Line File ")
@click.option('--loss',default='neg_log_loss',help="Model Evaluation Method ")
@click.option('--cross-validation',default=10, help="Cross Validation ")
def classification(method,pipe,cross_validation,loss):
    """
        this for select classification model
    """

    if pipe == None:
        click.echo(click.style("[X] You need import your data",fg='red',bg='black'))
        sys.exit(1)
    else:
        spec = importlib.util.spec_from_file_location(pipe[:-3],pipe)
        if spec is not None:
            datapipe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(datapipe)
            if hasattr(datapipe, "itrain_pipe"):
                x_train, y_train, x_test, y_test = datapipe.itrain_pipe()
            else:
                click.echo(click.style("[X] Data load script was not expected, Please Check it Again",fg='red'))
  
    if os.path.isfile("".join(method)):
        """
            load train method from file, also need in debug file, so it's etl util ??
        """
        from sklearn.model_selection import cross_validate
        from sklearn import metrics
        try:
            loader = importlib.machinery.SourceFileLoader('Model', method)
            m = loader.load_module()
            clf = m.Model()
            clf.fit(x_train, y_train)
            modeldumpname = "saved/{}-{}.pkl".format(os.path.basename(method), time.ctime().replace(" ","-"))
            dumpit(clf, modeldumpname)
            
            print("Training Score: \t",clf.score(x_test, y_test), end="")
            if loss in metrics.__all__:
                loss = getattr(metrics, loss)
            else:
                loss = metrics.log_loss
            y_prob = clf.predict_proba(x_test)
            y_predict = clf.predict(x_test)
            
            print("Classification Report: ", metrics.classification_report(y_test,y_predict))
            print("{}:{}".format(loss.__name__,loss(y_test, y_prob)))
        
        except Exception as e:
            print(" Exception: {}".format(e))

    else:
        methods = [CLASSFICATIONS.get(x) for x in method if CLASSFICATIONS.get(x)]
        if methods:
            clfs = Model(methods,loss=loss)
            clfs.fit(x_train,y_train=y_train,x_test=x_test, y_test=y_test)
            clfs.save()
        else:
            click.echo(click.style("[!] Now We Will Use Default All Method", fg="green",bg="black"))
            clfs = Model(CLASSFICATIONS.values(),loss=loss)
            clfs.fit(x_train,y_train=y_train,x_test=x_test, y_test=y_test)
            clfs.save()

def main():
    cli()

if __name__ == '__main__':
    main()