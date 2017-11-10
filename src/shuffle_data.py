'''
Reshuffles data into train and validation directories
'''

import argparse
import os
from os.path import isdir, isfile, join
from os import listdir
import shutil
import random

'''
data file structure is the following:

<path>/
    all/
    train/
    validation/

'''
def shuffle_dir(path, frequency):
    all_path = join(path, "all")
    train_dir = join(path, "train")
    val_dir = join(path, "validation")

    try:
        print("[SHUFFLE DATA] Clearing train directory...")
        shutil.rmtree(train_dir)
    except:
        print("[SHUFFLE DATA] No train directory found")

    try:
        print("[SHUFFLE DATA] Clearing validation directory...")
        shutil.rmtree(val_dir)
    except:
        print("[SHUFFLE DATA] No validation directory found")

    os.mkdir(train_dir)
    os.mkdir(val_dir)

    class_dirs = [(join(all_path, f),f) for f in listdir(all_path) if isdir(join(all_path, f))]

    for (classpath, classname) in class_dirs:
        train_save_dir = join(train_dir, classname)
        val_save_dir = join(val_dir, classname)
        os.mkdir(train_save_dir)
        os.mkdir(val_save_dir)

        for (imagepath, imagename) in [(join(classpath, f), f) for f in listdir(classpath)]:
            i = random.uniform(0, 1)
            if frequency >= i: #frequency is probability we assignt to train
                shutil.copy2(imagepath, join(train_save_dir, imagename))
            else:
                shutil.copy2(imagepath, join(val_save_dir, imagename))

    print("[SHUFFLE DATA] Finished shuffling data")

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Generate preprocessed bottleneck data')
    parser.add_argument('--data_dir', '-d', required = True,
        help='Input folder path. Class directories should be stored in data/all/')
    parser.add_argument('--frequency', '-f', required = True,
        help='Proportion of data to be put into train set')
    args = parser.parse_args()

    if not isdir(args.data_dir):
        raise Exception("Directory does not exist: " + args.data_dir)

    shuffle_dir(args.data_dir, float(args.frequency))
