'''
Secondary script to begin training active learning model
'''

import argparse
import os
from os.path import isdir, isfile, join
from os import listdir
from remove_DS_Store import *
import random
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
import pickle as cPickle

'''
data_dir, as always, should be a directory with the minimum structure:
<data_dir>/
    all/
        class0/
            img_class0.jpg
            ...
        class1/
            img_class1.jpb
            ...
        .../
This function will return list of tuples of (absolute_img_path, label)
'''
def get_image_list(data_dir):
    # remove all .DS_Store files first
    nuke_current_dir()
    data_dir = join(data_dir, "all")
    image_dirs = [(join(data_dir, f),f) for f in listdir(data_dir) if isdir(join(data_dir, f))]
    images = []
    classes = {}
    class_index = 0
    for (img_path, label) in image_dirs:
        if label not in classes:
            classes[label] = class_index
            class_index += 1
        images += [(join(img_path, f),classes[label]) for f in listdir(img_path)]
    random.shuffle(images)
    return (images, classes)


def present_image(path):
    frame = cv2.imread(path)
    cv2.imshow('frame', frame)
    while True:
        ch = cv2.waitKey()
        # this is hardcoded for SEA LION
        if ch & 0xFF == ord('a'):
            return 'sealion'
        # this is hardcoded for DARTH VADER
        if ch & 0xFF == ord('l'):
            return 'darthvader'

'''
Centerpiece to our entire project. Should be very robust in the future,
and have good sampling techniques
'''
def get_query_data(features, labels, i):
    # TODO: implement better selection heuristic
    feature_slice = features[i*10:i*10+10]
    label_slice = labels[i*10:i*10+10]

    result = []
    for i, feature in enumerate(feature_slice):
        result.append((feature, labels[i][0]))
    return result

# we take in a massive list of images(shuffled), and we save their feature data
def create_bottleneck_features(images):
    model = VGG16(weights='imagenet', include_top=False)
    features = []
    for (img_path, label) in images:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        features.append(feature[0].ravel())

    print("Saving image data")
    np.save(join(os.getcwd(), "weights/al_features.npy"), np.array(features))
    np.save(join(os.getcwd(), "weights/al_labels.npy"), images)

def train_classifier(data_dir):
    # step 0. initialize our model
    clf = LinearSVC(multi_class="ovr", loss='hinge', verbose=True, max_iter=250, C=10**5)
    # step 1. get list of all images
    (images,classes) = get_image_list(data_dir)


    # if True: #check if we are missing files OR we want to override them
    #     print("Creating bottleneck features")
    #     create_bottleneck_features(images)

    # bottleneck features is array of processed images
    bottleneck_features = np.load(join(os.getcwd(), "weights/al_features.npy"))
    # bottleneck labels is array of [image path, label] arrays
    bottleneck_labels = np.load(join(os.getcwd(), "weights/al_labels.npy"))

    unlabeled_data = bottleneck_labels

    # step 2. pick ten random images to initially manually label,
    # put them into some list
    X = []
    y = []
    for i in range(10):
        print("Querying on iteration:", i)
        # get_query_data will look through these features, and based
        # on "some" heuristic (random for now), it will return the
        # processed features, its index, and image path
        query = get_query_data(bottleneck_features, unlabeled_data, i)
        # manually label our current data
        for (features, img_path) in query:
            manual_label = classes[present_image(img_path)]
            X.append(features)
            y.append(manual_label)

        print("Fitting model..")
        clf.fit(X,y)

        #TODO: Evaluate periodic accuracy of our svm model with the rest of our
        # unlabeled data!!


if __name__=='__main__':
    '''
    First, we vectorize our images into nd arrays, along with their
    corresponding labels
    We store this as our source of truth

    We first randomly(?) pick 10 random vectors to label

    (*)Treat this as our current feature, label dataset.
        -> train up our model using this X, y dataset. (Default=SVM)

    With this new model, rerun on our entire dataset,
        -> based on some heuristic (deafult=random), select top 10 features

    Label these 10 features, append them to X' = X U {features}, Y' = Y U {labels}

    Rinse and repeat until 100 labels (go back to step *)
    '''
    train_classifier(join(os.getcwd(), "data"))
