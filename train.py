from models import Model, load_classifications
import importlib
import sys
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
@click.option('--method',default='allmethod',help="Your method for training model", multiple=True)
@click.option('--pipe',default=None,help="Data Pipe Line File ")
@click.option('--cross-validation',default=10, help="Cross Validation ")
def classification(method,pipe,cross_validation):
    """
        this for select classification model
    """

    if pipe == None:
        click.echo(click.style("[X] You need import your data"))
        sys.exit(1)
    else:
        spec = importlib.util.spec_from_file_location(pipe[:-3],pipe)
        if spec is not None:
            datapipe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(datapipe)
            if hasattr(datapipe, "iload_pipe"):
                x_train, y_train, x_test, y_test = datapipe.iload_pipe()
            else:
                click.echo(click.style("[X] Data load script was not expected, Please Check it Again",fg='red'))

    methods = [CLASSFICATIONS.get(x) for x in method if CLASSFICATIONS.get(x)]
    if methods:
        clfs = Model(methods)
        clfs.fit(x_train,y_train=y_train,x_test=x_test, y_test=y_test)
        clfs.save()
    else:
        click.echo(click.style("[!] Now We Will Use Default All Method", fg="green",bg="black"))
        clfs = Model(CLASSFICATIONS.values())
        clfs.fit(x_train,y_train=y_train,x_test=x_test, y_test=y_test)
        clfs.save()

def main():
    cli()

if __name__ == '__main__':
    main()