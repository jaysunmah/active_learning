'''
Train SVM file.
This script should take in bottleneck files, train an svm on them, and
save the svm
'''

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
import pickle as cPickle
import argparse

def v0_svm():
    clf = LinearSVC(multi_class="ovr", loss='hinge', verbose=True, max_iter=250, C=10**5)
    return clf

'''
Creates and fits an svm based on the training and validtion data
'''
def train_svm_impl(train_feature_file, train_label_file, val_feature_file, val_label_file, classes):
    model = v0_svm()
    train_features = np.load(train_feature_file)
    train_labels = np.load(train_label_file)
    print("[TRAIN SVM] Fitting model")
    model.fit(train_features, train_labels)
    print("\n[TRAIN SVM] Done fitting model")

    val_features = np.load(val_feature_file)
    val_labels = np.load(val_label_file)

    pred = model.predict(val_features)

    confusion = np.zeros((classes,classes))
    for (i, guess) in enumerate(pred):
        confusion[val_labels[i]][guess] += 1

    sum_rows = np.sum(confusion, axis=1)
    confusion = confusion / sum_rows

    print(confusion)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train SVM model based on preprocessed features')
    parser.add_argument('--train_feature', required = True,
        help='Location of training features')
    parser.add_argument('--train_label', required = True,
        help='Location of training labels')
    parser.add_argument('--val_feature', required = True,
        help='Location of validation fetures')
    parser.add_argument('--val_label', required = True,
        help='Location of validation labels')
    args = parser.parse_args()

    train_svm_impl(args.train_feature, args.train_label, args.val_feature, args.val_label)
