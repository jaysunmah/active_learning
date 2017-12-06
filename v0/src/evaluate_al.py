'''
Evaluates our models by feeding in a movie and it will return
'''
import time
import argparse
import os
from os.path import isdir, isfile, join
from os import listdir
from remove_DS_Store import *
import random
import cv2
import math
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import preprocessing
import pickle as cPickle

def evaluate(query_method, input_file):
    with open(join(os.getcwd(), "weights/" + query_method + "-clf.pkl"), 'wb') as fid:
        clf = cPickle.load(fid)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Script for evaluating success of classifier')
    parser.add_argument('--input_file', '-i', required=True,
        help='Input file (.mov format)')
    parser.add_argument('--query_method', '-q', default='random',
        help='Query method to evluate')
    args = parser.parse_args()

    evaluate(args.query_method, args.input_file)
