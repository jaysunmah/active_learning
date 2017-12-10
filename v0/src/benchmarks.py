'''
This file will test and evaluate all of our current selection techniques,
and display them (ideally) on a single graph
'''
import os
from os.path import isdir, isfile, join
from os import listdir
import argparse
import matplotlib.pyplot as plt
import numpy as np

import train_al

def run_benchmark(iterations=10,batch_size=10):
    clfs = []
    xaxis = [(i + 1) * batch_size for i in range(iterations)]
    queries = ["random", "uncertainty", "entropy", "ceal"]
    for query in queries:
        #TODO: Have train_classifier develop an image result which displays
        # the confusions?
        iters = 20
        avg = 0
        avg_acc = np.zeros(iterations)
        for i in range(iters):
            (acc,clf) = train_al.train_classifier(join(os.getcwd(), "data"), False, iterations, batch_size, query, False, False)
            avg_acc += np.array(acc)
            avg += acc[-1]
        avg_acc /= iters
        print("[BENCHMARK]", query, avg / iters)
        plt.plot(xaxis, avg_acc)
    plt.legend(queries)
    plt.xlabel("Labeled Samples")
    plt.ylabel("% Accuracy")
    plt.title("Active Learning Selection Accuracies")
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Testing benchmarks for various active learning seleciton techniques')
    run_benchmark()
