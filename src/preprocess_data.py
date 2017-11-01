'''
Generate bottleneck features and labels for training
This script takes in an overall data directory containing all of the
images. Will end up saving .npy files which can be reused for training later
'''

import argparse
import os
from os.path import isdir, isfile, join
from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

TRAIN_FEATURES_FILE = 'train_features.npy'
TRAIN_LABELS_FILE = 'train_labels.npy'

VAL_FEATURES_FILE = 'val_features.npy'
VAL_LABELS_FILE = 'val_labels.npy'

'''
data file structure is the following:

<path>/
    all/
    train/
    validation/

'''

def get_bottleneck_data(path):
    train_dir = join(path, "train")
    val_dir = join(path, "validation")
    print("Obtaining bottleneck data")
    model = VGG16(weights='imagenet', include_top=False)

    for (path, type) in [(train_dir,'train'), (val_dir,'validation')]:
        class_dirs = [(join(path, f),f) for f in listdir(path) if isdir(join(path, f))]
        bottleneck_features = []
        bottleneck_labels = []

        i = 0
        classes = {}

        for (classpath, classname) in class_dirs:
            if classname not in classes:
                classes[classname] = i
                i += 1

            #TODO PROCESS THESE IN BATCHES
            print("Processing", classpath)
            for img_path in [join(classpath, img) for img in listdir(classpath)]:
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = model.predict(x)

                bottleneck_features.append(features[0].ravel())
                bottleneck_labels.append(classes[classname])

        bottleneck_features = np.array(bottleneck_features)
        bottleneck_labels = np.array(bottleneck_labels)

        print('Saving',type,'features...')
        np.save(type + "_features.npy", bottleneck_features)
        np.save(type + "_labels.npy", bottleneck_labels)
        print(type, 'features saved!')


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Generate preprocessed bottleneck data')
    parser.add_argument('--data_dir', '-d', required = True,
        help='Input folder path. Directory should contain set of directories whose name correspodns to the images labels')
    args = parser.parse_args()

    if not isdir(args.data_dir):
        raise Exception("Directory does not exist: " + args.data_dir)

    get_bottleneck_data(args.data_dir)
