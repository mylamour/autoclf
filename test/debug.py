import click
import os,sys

sys.path.append(os.getcwd())

import tempfile

try:
    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.ipapp import load_default_config
except ImportError:
    from IPython.frontend.terminal.embed import InteractiveShellEmbed
    from IPython.frontend.terminal.ipapp import load_default_config


BANNER = """

        .------..------..------..------..------..------..------.
        |A.--. ||U.--. ||T.--. ||O.--. ||C.--. ||L.--. ||F.--. |
        | (\/) || (\/) || :/\: || :/\: || :/\: || :/\: || :(): |
        | :\/: || :\/: || (__) || :\/: || :\/: || (__) || ()() |
        | '--'*|| '--'D|| '--'E|| '--'B|| '--'U|| '--'G|| '--'*|
        `------'`------'`------'`------'`------'`------'`------'

"""

def train():
    pass

def test():
    pass

@click.command()
@click.option("--load", "-l", multiple=True, default=None, help="Your file need to be debug")
# @click.option("--mode", "-m", multiple=True, default=None, help="Choose classes or function from file")
def debug(load):
    shell = InteractiveShellEmbed.instance(banner1=BANNER,  user_ns={})
    if load == None:
        click.echo("[!] You didn't select any file to debug, we only lanuch a ipython shell")
        return shell()
    click.echo("Your debug file is {}: \n Debug shell would start automaticly".format(load))
    
    with tempfile.NamedTemporaryFile(delete=False) as wfd:

        for _ in load:
            # print(_)
            if os.path.isfile(_):
                    # %load -s Class, function filename
                with open(_,'rb') as fd:                        # if not 'rb', "would get byter-like not str error"
                    # shutil.copyfileobj(fd,wfd,1024*1024*10)   # But not for tempfile
                    wfd.write(fd.read())
                    wfd.write('\n'.encode())                    # if not, you would get some error probably
            else:
                click.echo("[X] Error: {} loading failed, FILE NOT EXIST".format(_))
                sys.exit(1)

    shell.run_line_magic(magic_name="load", line=wfd.name)

    return shell()
                
    # methods = 'models.py'

    # shell.run_line_magic(magic_name="load", line=methods)

    #Todo: echo load function, class, and so on
    #Todo: Get your basic data

def main():
    debug()

if __name__ == '__main__':
    main()