import pandas as pd
import numpy as np

import importlib
import click

import os
import sys
import glob
import csv

from sklearn.externals import joblib

def predict_model(model,pdata,out=None):
    p = None
    proba = None
    folder = "saved"
    model = joblib.load(model)
    mname = model.__class__.__name__
    click.echo(click.style(' predict use model: {}'.format(mname)))
    
    if out:
        if os.path.exists(out) and os.path.isdir(out):
            pass            
        else:
            os.mkdir(out)
        
        folder = out

        # np.savetxt("{}.predict".format(out), proba, delimiter=",")
        # np.savetxt("{}.proba".format(out), p, delimiter=",")
    
    if hasattr(model,'predict'):
        p = model.predict(pdata)
        np.savetxt("{}/{}.predict".format(folder,mname), p, delimiter=",")
    
    if hasattr(model,'predict_proba'):
        proba = model.predict_proba(pdata)
        np.savetxt("{}/{}.proba".format(folder,mname), proba, delimiter=",")
        
        # res = pd.DataFrame(p,columns=["predict"])
        # res.to_csv(out,index=False)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--method',default=None,help="Your method for training model", multiple=True)
@click.option('--pipe',default=None,help="Data Pipe Line File ")
@click.option('--out', default=None,help="Directory for save predict ", required=False)
def predict(pipe, method, out):
    if pipe == None:
        click.echo(click.style("[X] You need import your data", fg='red',bg='black'))
        sys.exit(1)
    else:
        spec = importlib.util.spec_from_file_location(pipe[:-3],pipe)
        if spec is not None:
            datapipe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(datapipe)
            if hasattr(datapipe, "ipredict_pipe"):
                x_train, y_train, x_test, y_test = datapipe.iload_pipe()
            else:
                click.echo(click.style("[X] Data load script was not expected, Please Check it Again",fg='red'))

    saved = os.path.join(os.path.dirname(os.path.realpath(__file__)),method[0])
    if os.path.isdir(saved):
        click.echo(click.style("Use Batch Models From {}".format(saved),fg='green',bg='black'))
        
        models = glob.glob("{}/*.pkl".format(saved))
        with click.progressbar(models) as models:
            for model in models:
                try:
                    predict_model(model,x_test,out)
                except Exception as e:
                    print("[X] {} Error".format(model))
    else:
        with click.progressbar(method) as methods:            
            for model in methods:
                try:
                    predict_model(model,x_test,out)
                except Exception as e:
                    print("[X] {} Error".format(model))


@cli.command()
@click.option('--method',default='allmethod',help="Your method for training model", multiple=True)
@click.option('--pipe',default=None,help="Data Pipe Line File ")
def predict_proba():
    pass
            
def main():
    cli()

if __name__ == '__main__':
    main()