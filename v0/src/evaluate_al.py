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

from video_to_frames import convert_video

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import preprocessing
import pickle as cPickle



def evaluate(query_method, input_file, recreate_bottleneck, convert_frames):
    '''
    Step 1: convert video into frames, load them into tmp folder
    '''
    source_dir = join(os.getcwd(), "tmp")
    if convert_frames:
        print("[EVAL] Parsing video into frames")
        convert_video(join(os.getcwd(), input_file), source_dir, "eval", 2)

    '''
    Step 2: in our source directory, look at each image, process it under our vgg16 model,
    and extract them features
    '''
    images = [join(source_dir, f) for f in listdir(source_dir)]
    if recreate_bottleneck:
        print("[EVAL] Extracting frame features")
        from keras.applications.vgg16 import VGG16
        from keras.preprocessing import image
        from keras.applications.vgg16 import preprocess_input
        model = VGG16(weights='imagenet', include_top=False)

        imsize = 224
        features = np.zeros((len(images), 7 * 7 * 512))
        x_batch = np.zeros((len(images),imsize,imsize,3))

        for (i,img_path) in enumerate(images):
            img = image.load_img(img_path, target_size=(imsize, imsize))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x_batch[i] = x[0]

        print("[EVAL] Creating features")
        batch_features = model.predict(x_batch)

        for (i, feature) in enumerate(batch_features):
            features[i] = feature.ravel()
        np.save(join(os.getcwd(), "weights/evaluate_features.npy"), np.array(features))
    features = np.load(join(os.getcwd(), "weights/evaluate_features.npy"))

    print("[EVAL] Classifying features with svm")
    clf_file = join(os.getcwd(), "weights/" + query_method + "-clf.pkl")
    with open(clf_file, 'rb') as fid:
        clf = cPickle.load(fid)

    probs = clf.predict_proba(features)
    classes = np.argmax(probs, axis=1)
    prob_classes = np.max(probs, axis=1)

    zipped_probs = [(i,p,c) for (i,(p,c)) in enumerate(zip(prob_classes, classes))]
    # print(zipped_probs)

    good_pics = [i for (i,p,c) in sorted(zipped_probs, key=lambda x: x[1], reverse=True) if c == 1]

    '''
    Final step: Stitch together our best photos here
    Reduce icon size to 128px by 128px?
    '''

    # image padding
    impadding = 15
    imsize = 128
    nrows = 5
    ncols = 6

    from keras.preprocessing import image

    result = np.zeros((nrows*(2*impadding+imsize), ncols*(2*impadding+imsize), 3), np.uint8)
    result.fill(255)
    print("[EVAL] Stitching images together")

    icon_count = 0
    for i in good_pics[:nrows * ncols]:
        img = image.load_img(images[i], target_size=(imsize, imsize))
        x = image.img_to_array(img)
        row = (icon_count//ncols) * (imsize + 2*impadding) + impadding
        col = (icon_count%ncols) * (imsize + 2*impadding) + impadding

        for r in range(imsize):
            for c in range(imsize):
                for d in range(3):
                    result[row+r,col+c,d] = x[r,c,3-(d+1)]

        icon_count += 1

    print("[EVAL] DONE! Saving result")
    cv2.imwrite(join(os.getcwd(), "weights/" + query_method + "-res.jpg"), result)
    cv2.imshow('result', result)
    while True:
        ch = cv2.waitKey()
        if ch & 0xFF == ord('q'):
            return

if __name__=='__main__':
    #TODO Set flag to convert video frames andalso recreate bottleneck features
    parser = argparse.ArgumentParser(description='Script for evaluating success of classifier')
    parser.add_argument('--input_file', '-i', required=True,
        help='Input file (.mov format)')
    parser.add_argument('--query_method', '-q', default='random',
        help='Query method to evluate')
    parser.add_argument('--recreate_bottleneck', '-r', default='0',
        help='Flag to recreate bottleneck features')
    parser.add_argument('--convert_frames', '-c', default='0',
        help='Flag to convert video into frames')
    args = parser.parse_args()

    recreate_bottleneck = args.recreate_bottleneck == '1'
    convert_frames = args.convert_frames == '1'

    evaluate(args.query_method, args.input_file, recreate_bottleneck, convert_frames)
