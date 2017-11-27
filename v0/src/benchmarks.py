'''
This file will test and evaluate all of our current selection techniques,
and display them (ideally) on a single graph
'''
import os
from os.path import isdir, isfile, join
from os import listdir
import argparse

import train_al

def run_benchmark(iterations=10,batch_size=10):
    for query in ["random", "uncertainty", "entropy","ceal"]:
        #TODO: Have train_classifier develop an image result which displays
        # the confusions?
        acc = train_classifier(join(os.getcwd(), "data"), False, iterations, batch_size, query)
        '''
        TODO: append these accuracies with their query method,
        plot them on matplotlib,
        '''

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Testing benchmarks for various active learning seleciton techniques')
    run_benchmark()
