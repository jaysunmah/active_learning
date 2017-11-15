'''
This script will nuke all .DS_Store files recursively, starting from
the directory from which it was called
'''

import os
from os.path import isdir, isfile, join
from os import listdir

def remove_dsstore(path):
    dirs = [join(path,d) for d in listdir(path) if isdir(join(path,d))]
    if os.path.exists(join(path, ".DS_Store")):
        os.remove(join(path,".DS_Store"))
    for dir in dirs:
        remove_dsstore(dir)
    return


def nuke_current_dir():
    remove_dsstore(os.getcwd())
