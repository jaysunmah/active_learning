'''
This script stores all of our sampling techniques.
Query needs to be of the form
[(feature, image_path, src_truth_label), ...]
'''

import numpy as np
import aloss
import math
import random
import copy
import time

# we will evenly select batch_size/2 of each class
# WARNING! This assumes we only have two classes to consider
def init_select(clf,unlabeled_data,labels,batch_size):
    query_indices = set()
    c1 = math.floor(batch_size) / 2
    c2 = math.ceil(batch_size) / 2
    counts = [c1, c2]
    classes = {}
    i = 0
    while len(query_indices) != batch_size:
        if labels[i][1] not in classes:
            classes[labels[i][1]] = len(classes)
        if counts[classes[labels[i][1]]] > 0:
            counts[classes[labels[i][1]]] -= 1
            query_indices.add(i)
        i += 1
    result = [(unlabeled_data[i], labels[i][0], int(labels[i][1])) for i in query_indices]
    for i in query_indices:
        unlabeled_data[i][0] = -1
        labels[i] = None
    # unlabeled_data = np.array([d for (i,d) in enumerate(unlabeled_data.tolist()) if i not in query_indices])
    # labels = np.array([d for (i,d) in enumerate(labels.tolist()) if i not in query_indices])
    unlabeled_data = [d for d in unlabeled_data if d[0] != -1]
    labels = [d for d in labels if d is not None and d[0] != 'None']

    return (result, unlabeled_data, labels)


# random sampling: will return batch_size random samples
def random_select(clf, unlabeled_data, labels, batch_size):
    if len(unlabeled_data) <= batch_size:
        query_indices = list(range(len(unlabeled_data)))
        result = [(copy.deepcopy(unlabeled_data[i]), labels[i][0], int(labels[i][1])) for i in query_indices]
        return result, [], []

    query_indices = set()
    while len(query_indices) != batch_size:
        i = random.randrange(0, len(unlabeled_data))
        if i not in query_indices:
            query_indices.add(i)

    result = [(unlabeled_data[i], labels[i][0], int(labels[i][1])) for i in query_indices]
    for i in query_indices:
        unlabeled_data[i][0] = -1
        labels[i] = None
    # unlabeled_data = np.array([d for (i,d) in enumerate(unlabeled_data.tolist()) if i not in query_indices])
    unlabeled_data = [d for d in unlabeled_data if d[0] != -1]
    labels = [d for d in labels if d is not None and d[0] != 'None']

    return (result, unlabeled_data, labels)

# uncertainty sampling: takes the most uncertain data and returns it
def uncertainty(clf,unlabeled_data,labels,batch_size):
    uncertainties = aloss.instance_uncertainties(clf, unlabeled_data)
    zipped = [(i, d) for (i, d) in enumerate(uncertainties)]
    sorted_uncertanties = sorted(zipped,key=lambda x: x[1], reverse=True)
    query_indices = [i for (i, d) in sorted_uncertanties[:batch_size]]

    result = [(unlabeled_data[i], labels[i][0], int(labels[i][1])) for i in query_indices]

    for i in query_indices:
        unlabeled_data[i][0] = -1
        labels[i] = None

    unlabeled_data = [d for d in unlabeled_data if d[0] != -1]
    labels = [d for d in labels if d is not None and d[0] != 'None']

    return (result, unlabeled_data, labels)

# optimal subset selection:
# http://ieeexplore.ieee.org/document/6272387/
def aloss_select(clf,unlabeled_data,labels,batch_size):
    # compute our M matrix
    n = len(unlabeled_data)
    M = np.zeros((n,n))
    uncertainties = aloss.instance_uncertainties(clf, unlabeled_data)
    now = time.time()
    # print("Computing disparities, may take a while")
    disparities = aloss.instance_disparities(unlabeled_data)
    # print("Finished computing disparities in", time.time() - now)
    disparities = disparities / np.max(disparities)
    for i in range(n):
        disparities[i][i] = uncertainties[i]

    # this will take the matrix, sum it across its rows, and return the
    # top batch_size samples with the highest sums
    query_indices = aloss.greedy_solver(disparities, batch_size)
    # print("Brute force solving...")
    # now = time.time()
    # query_indices = aloss.brute_force_solver(disparities, batch_size)
    # print("Finished computing brute force in", time.time() - now)

    result = [(unlabeled_data[i], labels[i][0], int(labels[i][1])) for i in query_indices]

    queried_indices = set(query_indices)
    unlabeled_data = np.array([d for (i,d) in enumerate(unlabeled_data) if i not in queried_indices])
    labels = np.array([d for (i,d) in enumerate(labels) if i not in queried_indices])
    return (result, unlabeled_data, labels)

#https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Li_Adaptive_Active_Learning_2013_CVPR_paper.pdf
def entropy(clf,unlabeled_data,labels,batch_size):
    probs = clf.predict_proba(unlabeled_data)
    fs = []

    # iterates through each set of probability distributions
    for distribution in probs:
        total = 0
        for p in distribution:
            total += p * math.log(p)
        fs.append(-1 * total)
    fs = list(enumerate(fs))
    fs = sorted(fs, key=lambda x: x[1], reverse=True)
    query_indices = [i for (i, d) in fs[:batch_size]]

    result = [(unlabeled_data[i], labels[i][0], int(labels[i][1])) for i in query_indices]

    queried_indices = set(query_indices)
    unlabeled_data = np.array([d for (i,d) in enumerate(unlabeled_data) if i not in queried_indices])
    labels = np.array([d for (i,d) in enumerate(labels) if i not in queried_indices])
    return (result, unlabeled_data, labels)

'''
t0 = initial threshold
dr = decay rate, per iteraton
tries to pseudo label datapoints based on highest predctions (above our
uncertainty threshold)
'''
def get_pseudo_labels(clf,unlabeled_data,labels,batch_size,iter,dr,t0):
    t = t0 - iter * dr
    probs = clf.predict_proba(unlabeled_data)
    # the predicted classes of our features
    pred_classes = np.argmax(probs, axis=1)
    # the probability of those classes
    prob_classes = np.max(probs, axis=1)
    # we want to zip the data in the following order:
    # (index, probability, class), sort by probability
    data = [(p,i,c) for (i,(c,p)) in enumerate(zip(pred_classes, prob_classes))]
    data = sorted(data, key=lambda x: x[0], reverse=True)
    data = [(i, c) for (p,i,c) in data[:batch_size] if p >= t]

    pseudo_labels = [(unlabeled_data[i],c) for (i,c) in data]

    query_indices = [i for (i,c) in data]
    for (i, c) in data:
        unlabeled_data[i][0] = -1
        labels[i] = None

    unlabeled_data = [d for d in unlabeled_data if d[0] != -1]
    labels = [d for d in labels if d is not None and d[0] != 'None']

    return (pseudo_labels,unlabeled_data,labels)


# cost effective active learning
# https://arxiv.org/pdf/1701.03551.pdf
'''
Currently, this is sampling is simply based on uncertainty. Our addition from
ceal method is the fact that we also pseudo label datapoints, alongside our
manual labels
'''
def ceal(clf,unlabeled_data,labels,batch_size):
    uncertainties = aloss.instance_uncertainties(clf, unlabeled_data)
    zipped = list(enumerate(uncertainties))
    sorted_uncertanties = sorted(zipped,key=lambda x: x[1], reverse=True)
    query_indices = [i for (i, d) in sorted_uncertanties[:batch_size]]

    result = [(unlabeled_data[i], labels[i][0], int(labels[i][1])) for i in query_indices]

    queried_indices = set(query_indices)
    unlabeled_data = np.array([d for (i,d) in enumerate(unlabeled_data.tolist()) if i not in queried_indices])
    labels = np.array([d for (i,d) in enumerate(labels.tolist()) if i not in queried_indices])
    return (result, unlabeled_data, labels)
