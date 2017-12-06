'''
Secondary script to begin training active learning model

Updates as of 11/25/17:
- Look into revamping how we are evaluating our model? The reason for this
is because we might be labeling things as "bad" when we originally labled them
as "good", but in reality they're pertty bad. (or vice versa).
- Display 40 random images deemed to be "good", and 40 random images deemed to be
"bad" after we are done with our entire net. Let's see qualitatively how we perform.
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

import aloss
import al_selection

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
def get_image_class(data_dir):
    data_dir = join(data_dir, "all")
    image_dirs = sorted([f for f in listdir(data_dir) if isdir(join(data_dir, f))])
    classes = {}
    for key in range(len(image_dirs)):
        classes[image_dirs[key]] = key
    return classes

def get_image_list(data_dir):
    # remove all .DS_Store files first
    nuke_current_dir()
    classes = get_image_class(data_dir)
    data_dir = join(data_dir, "all")
    image_dirs = [(join(data_dir, f),f) for f in listdir(data_dir) if isdir(join(data_dir, f))]
    images = []
    for (img_path, label) in image_dirs:
        images += [(join(img_path, f),classes[label]) for f in listdir(img_path)]
    random.shuffle(images)
    return images

'''
This function will present an image, and return
some "hardcoded" value for what class we manually
assign it.
'''
def present_image(path):
    frame = cv2.imread(path)
    cv2.imshow('frame', frame)
    while True:
        ch = cv2.waitKey()
        # this is hardcoded for SEA LION
        if ch & 0xFF == ord('0'):
            return 'bad'
        # this is hardcoded for DARTH VADER
        if ch & 0xFF == ord('1'):
            return 'good'

'''
Evaluates model in terms of performance with
source of truth to validate
'''
def evaluate_model(model, data, src_of_truth):
    preds = model.predict(data)
    labels = np.array([int(x) for (f, x) in src_of_truth])
    diffs = np.bitwise_xor(preds, labels)
    accuracy = 100 - (sum(diffs) / len(diffs) * 100)
    return accuracy


'''
Centerpiece to our entire project. Should be very robust in the future,
and have good sampling techniques
Query needs to be of the form
[(feature, image_path, src_truth_label), ...]
'''
def get_query(clf,unlabeled_data,labels,batch_size,query_method):
    if query_method == "init":
        return al_selection.init_select(clf,unlabeled_data,labels,batch_size)
    if query_method == 'random':
        return al_selection.random_select(clf,unlabeled_data,labels,batch_size)
    elif query_method == 'uncertainty':
        return al_selection.uncertainty(clf,unlabeled_data,labels,batch_size)
    elif query_method == 'aloss':
        return al_selection.aloss_select(clf,unlabeled_data,labels,batch_size)
    elif query_method == 'entropy':
        return al_selection.entropy(clf,unlabeled_data,labels,batch_size)
    elif query_method == 'ceal':
        # return al_selection.ceal(clf,unlabeled_data,labels,batch_size)
        return al_selection.uncertainty(clf,unlabeled_data,labels,batch_size)

'''
we take in a massive list of images(shuffled), and we save their feature data
Future improvement of this should be to display some progress bar, as this
can take a long time to compute
'''
def create_bottleneck_features(images):

    from keras.applications.vgg16 import VGG16
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input

    model = VGG16(weights='imagenet', include_top=False)
    features = np.zeros((len(images), 7 * 7 * 512))
    # features = np.zeros((len(images), 512))
    feature_index = 0
    total = len(images)
    curr = 0
    img_batch_size = 50
    # img_batch_size = 500
    imsize = 224
    # imsize = 32
    epochs = math.ceil(len(images) / img_batch_size)

    for epoch in range(epochs):
        print("Progress:", round(epoch / epochs * 100))
        begin = epoch * img_batch_size
        end = min((epoch + 1) * img_batch_size, len(images))

        x_batch = np.zeros((end-begin,imsize,imsize,3))
        i = 0

        for (img_path, label) in images[begin:end]:
            img = image.load_img(img_path, target_size=(imsize, imsize))
            # img = image.load_img(img_path, target_size=(32, 32))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            x_batch[i] = x[0]
            i += 1

        batch_features = model.predict(x_batch)
        # print(batch_features)
        # print(batch_features.shape)

        for feature in batch_features:
            features[feature_index] = feature.ravel()
            feature_index += 1
            # features.append(feature.ravel())
            curr += 1

    print("Progress: 100")
    print("Saving image data")
    # return
    np.save(join(os.getcwd(), "weights/al_features.npy"), np.array(features))
    np.save(join(os.getcwd(), "weights/al_labels.npy"), images)

def train_classifier(data_dir, reshuffle_data, iters, batch_size, query_method, verbose=True):
    '''
    step 0. initialize our model
    '''
    clf = LinearSVC(multi_class="ovr", loss='hinge', verbose=verbose)
    clf = CalibratedClassifierCV(clf)
    # clf = SVC(probability=True, verbose=True)

    '''
    step 1. get list of all images
            will shuffle images, but classes order should remain stable
    '''
    images = get_image_list(data_dir)
    classes = get_image_class(data_dir)
    if verbose: print("[TRAIN AL] CLASSES:", classes)

    '''
    step 2. process / retrieve all(?) feature data, if applicable
            after this step, we should have some list of unlabeled data,
            and some list of source of truths which we can use to lookup
            our "unlabeled" data when evaluating for accuracy
    '''

    #check if we are missing files OR we want to override them
    if not isfile(join(os.getcwd(), "weights/al_features.npy")) or reshuffle_data:
        if verbose: print("[TRAIN AL] Creating bottleneck features")
        create_bottleneck_features(images)

    # bottleneck features is array of processed images
    bottleneck_features = np.load(join(os.getcwd(), "weights/al_features.npy"))
    # bottleneck labels is array of [image path, label] arrays
    bottleneck_labels = np.load(join(os.getcwd(), "weights/al_labels.npy"))

    '''
    checkpoint 2.1: at this point, load up unlabeled data and src_of_truth lists.
            unlabeled_data, can be "undefined", but src_of_truth MUST be
            tuple of (img_path, class)
    '''
    unlabeled_data = bottleneck_features
    src_of_truth = bottleneck_labels

    '''
    step 3: for ITER iterations, with BATCH images in each batch,
            we query these image batches for manual labeling
            we append these new data + labels to my_data and my_labels
            which we will use to retrain our clf classifier (default=SVM)
    '''
    my_data = []
    my_labels = []
    accuracies = []
    for i in range(iters):
        if verbose: print("[TRAIN AL] Querying on iteration:", i)
        # get_query_data will look through these features, and based
        # on "some" heuristic (random for now), it will return the
        # the query, as well as the new unlabeled dataset and src of truth
        # query MUST have some image path zipped to end of tuple
        if len(my_data) == 0:
            (query,unlabeled_data,src_of_truth) = get_query(clf,unlabeled_data,src_of_truth,batch_size,"init")
        else:
            (query,unlabeled_data,src_of_truth) = get_query(clf,unlabeled_data,src_of_truth,batch_size,query_method)
            if query_method == 'ceal':
                # (pseudo_labels,unlabeled_data,src_of_truth) = al_selection.get_pseudo_labels(clf,unlabeled_data,src_of_truth,batch_size,i, 0.001,0.999)
                (pseudo_labels,unlabeled_data,src_of_truth) = al_selection.get_pseudo_labels(clf,unlabeled_data,src_of_truth,batch_size,i, 0.02,0.99)
                if verbose: print("[CEAL] adding", len(pseudo_labels), "data points")
                for (f, c) in pseudo_labels:
                    my_data.append(f)
                    my_labels.append(c)
                if verbose: print("[CEAL] data size:", len(my_labels))

        # manually label our current data
        for (features, img_path, true_label) in query:
            # manual_label = classes[present_image(img_path)]
            manual_label = true_label
            my_data.append(features)
            my_labels.append(manual_label)

        # Fit our classifier with our labeled data
        clf.fit(np.array(my_data),np.array(my_labels))

        # Evaluate periodic accuracy of our svm model with the rest of our
        # unlabeled data
        acc = evaluate_model(clf, unlabeled_data, src_of_truth)
        if verbose: print("\n[TRAIN AL]: Iteration=" + str(i) + " Model Accuracy:", str(acc) + "%")
        accuracies.append(acc)

    '''
    step 4: return / save our model!
    TODO: save our svm model in an output folder, as well as a confusion
    matrix visualization (show the images that we guessed wrong for each
    class.)
    '''
    if verbose: print("[TRAIN AL]", accuracies)

    with open(join(os.getcwd(), "weights/" + query_method + "-clf.pkl"), 'wb') as fid:
        cPickle.dump(clf, fid)

    return (accuracies, clf)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Main script for developing active learning model')
    parser.add_argument('--reshuffle_data', '-r', default=0, type=int,
        help='Flag to reshuffle data and reprocess bottleneck features and labels')
    parser.add_argument('--iterations', '-i', default=10, type=int,
        help='How many iterations we want to run')
    parser.add_argument('--batch_size', '-b', default=10, type=int,
        help='How large we want our batch sizes to be')
    parser.add_argument('--query_method', '-q', default='random',
        help='Query Selection')

    args = parser.parse_args()
    reshuffle_flag = args.reshuffle_data == 1

    train_classifier(join(os.getcwd(), "data"), reshuffle_flag, args.iterations, args.batch_size, args.query_method)
