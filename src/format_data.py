'''
Simple helper function that will format our data folder images so that
they are of the format <CLASS>_<UID>.jpg

Can be useful to keep things consistent later on
'''

import argparse
import os
from os.path import isdir, isfile, join
from os import listdir

def format_dir(path):
    classes = [(join(path, f),f) for f in listdir(path) if isdir(join(path, f))]
    for (dirpath, dirname) in classes:
        for (i, f) in enumerate(listdir(dirpath)):
            os.rename(join(dirpath, f), join(dirpath, dirname + "_" + str(i) + ".jpg"))
    print("Done!")

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Convert images to well formatted names')
    parser.add_argument('--input_dir', '-i', required = True,
        help='Input folder path. Directory should contain set of directories whose name correspodns to the images labels')
    args = parser.parse_args()

    if not isdir(args.input_dir):
        raise Exception("Directory does not exist: " + args.input_dir)

    format_dir(args.input_dir)
