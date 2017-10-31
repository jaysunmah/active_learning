'''
Generate bottleneck features and labels for training
'''

import argparse
import os
from os.path import isdir, isfile, join
from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

BOTTLENECK_FEATURES_FILE = 'bottleneck_features.npy'
BOTTLENECK_LABELS_FILE = 'bottleneck_labels.npy'

def get_bottleneck_data(path):
    print("Obtaining bottleneck data")
    model = VGG16(weights='imagenet', include_top=False)

    class_dirs = [(join(path, f),f) for f in listdir(path) if isdir(join(path, f))]
    bottleneck_features = []
    bottleneck_labels = []

    i = 0
    classes = {}

    for (classpath, classname) in class_dirs:
        if classname not in classes:
            classes[classname] = i
            i += 1

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

    print('Save data features...')
    np.save(BOTTLENECK_FEATURES_FILE, bottleneck_features)
    np.save(BOTTLENECK_LABELS_FILE, bottleneck_labels)
    print('Data features saved!')


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Generate preprocessed bottleneck data')
    parser.add_argument('--data_dir', '-d', required = True,
        help='Input folder path. Directory should contain set of directories whose name correspodns to the images labels')
    args = parser.parse_args()

    if not isdir(args.data_dir):
        raise Exception("Directory does not exist: " + args.data_dir)

    get_bottleneck_data(args.data_dir)
