'''
Main script to shuffle data, generate bottleneck features, and retrain svm
'''

import argparse
import os
from os.path import isdir, isfile, join
from os import listdir

import train_svm
import preprocess_data
import shuffle_data

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Shuffles data, generates bottleneck data, and trains svm')
    parser.add_argument('--data_dir', '-d', default='data/',
        help='Input folder path. Directory should contain set of directories whose name correspodns to the images labels')
    parser.add_argument('--frequency', '-f', default=0.7, type=int,
        help='Frequency of images to be shuffled into train directory')
    parser.add_argument('--gen_bottleneck_data', '-g', default=True, type=bool,
        help='Indicator to recreate bottleneck data')
    parser.add_argument('--weights_dir', default='weights/',
        help='Directory to save any metadata, such as models, weights, and features')
    parser.add_argument('--train_feature', default='train_features.npy')
    parser.add_argument('--train_label', default='train_labels.npy')
    parser.add_argument('--val_feature', default='validation_features.npy')
    parser.add_argument('--val_label', default='validation_labels.npy')
    args = parser.parse_args()

    for dir in [args.data_dir, args.weights_dir]:
        if not isdir(dir):
            raise Exception("Directory does not exist: " + dir)

    print("[TRAIN] Shuffling data directory: ", args.data_dir)
    shuffle_data.shuffle_dir(args.data_dir, args.frequency)

    gen_bottleneck_data = False or args.gen_bottleneck_data
    for f in [args.train_feature, args.train_label, args.val_feature, args.val_label]:
        if not isfile(join(args.weights_dir, f)):
            gen_bottleneck_data = True
            break

    if gen_bottleneck_data:
        print("[TRAIN] Generating bottleneck data")
        preprocess_data.get_bottleneck_data(args.data_dir,
            join(args.weights_dir, args.train_feature),
            join(args.weights_dir, args.train_label),
            join(args.weights_dir, args.val_feature),
            join(args.weights_dir, args.val_label))
    else:
        print("[TRAIN] Reusing existing bottleneck data")

    print("[TRAIN] Training svm")

    classes = len(listdir(join(args.data_dir, "all")))

    train_svm.train_svm_impl(join(args.weights_dir, args.train_feature),
        join(args.weights_dir, args.train_label),
        join(args.weights_dir, args.val_feature),
        join(args.weights_dir, args.val_label),
        classes)
