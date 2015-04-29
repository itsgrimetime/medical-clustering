import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
import operator

from os import listdir
from os.path import isdir

import time
import sys

def get_descriptors(img, sift, gray=False):
    if gray:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    return des

def get_folds(data, labels, k=7, shuffle=False):

    if len(data) != len(labels):
	print "Error: must have same number of data points and labels"
    fold_size = len(data) / fold

    if shuffle:
	shuf_index = range(len(data))
	random.shuffle(shuf_index)
	data = np.array([data[i] for i in shuf_index])
	labels = np.array([labels[i] for i in shuf_index])

    for chunk in range(kfold):
	val_data = array(data[chunk * fold_size : (chunk + 1) * fold_size])
	val_label = array(labels[chunk * fold_size : (chunk + 1) * fold_size])
	train_data = concat((data[0 : chunk * fold_size], data[(chunk + 1) * fold_size :]))
	train_label = concat((labels[0 : chunk * fold_size], labels[(chunk + 1) * fold_size :]))
	yield (val_data, val_label, train_data, train_label)

# TODO get labels matricies too
def get_filenames(directory):
    classes = {}
    count = 0
    for d in (x for x in listdir(data_dir) if not x.startswith('.')):
	classes[data_dir + d] = count
	count += 1
    print "{} classes".format(len(classes))
    for fldr in (f for f in listdir(data_dir) if isdir(data_dir + f)):
	for image_file in (f for f in listdir(data_dir + fldr) if f.split('.')[-1] == 'jpeg'):
	    full_path = './' + data_dir + fldr + '/' + image_file
	    # print full_path
	    yield (full_path, classes[data_dir + fldr])

def get_buckets(data, tup=False, pct=0.75, shuffle=False, kmeans=100):
    if pct < 1 or pct > 100:
	print "Error: pct must be between 1 and 100"
    if kmeans < 2:
	print "Error: kmeans must be grearter than 1"
    index = range(len(data))
    shuffle(index)
    desc = np.empty((0, 128), dtype=np.float32)
    if tup:
	desc = desc.vstack((desc, [data[i][0] for i in index[:(len(data) * pct)]]))
    else:
	dest = desc.vstack((desc, [data[i] for i in index[:(len(data) * pct)]]))

    temp, classified_points, means = cv2.kmeans(desc, K=kmeans, bestLabels=None,
	    criteria=crit, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
    return means

toolbar_width = 80
kmeans = 100
data_dir = "./../Training Data/"
sift = cv2.SIFT()
svm = cv2.SVM()
save_file = 'sift_save.npy'
data_save_file = 'data' + save_file
crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)

# setup toolbar
"""
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
"""

loaded_data = False
try:
    data = np.load(data_save_file)
    print "Loaded descriptors from {}".format(data_save_file)
    loaded_data = True
except IOError:
    print "Unable to load data from {}".format(data_save_file)
    data = []

if not loaded_data:
    for filename, label in get_filenames(data_dir):
	print "filename: {}, label: {}".format(filename, label)
	img = cv2.imread(filename)
	data.append((get_descriptors(img, sift), label))
    print "Saving {} descriptors to {}".format(len(data), data_save_file)
    np.save(data_save_file, data)

# temp, classified_points, means = cv2.kmeans(np.asarray(all_descriptors), K=2, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, kmeans), attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

# def get_buckets(data, tup=False, pct=0.75, shuffle=False, kmeans=100):

# get our defining "bucket of words"

means = get_buckets(data, tup=True, shuffle=True)

svm_params = dict(kernel_type = cv2.SVM_RBF,
	    svm_type = cv2.SVM_C_SVC,
	    C=1.0, gamma=0.1)

labels = [d[1] for d in data]
data_hists = []
for item, i in enumerate(data): # (img, label, SIFT descriptors) triple
    hist = np.zeros((0, len(buckets)))
    for desc in item[0]: # for each descriptor set each descriptor has 128 values)
	dists = []
	for mean, mindex in enumerate(means):
	    dists.append((mindex, np.linalg.norm(mean - desc)))
	dists = sorted(dists, key=lambda entry: entry[1])
	hist[dists[0][0]] += 1
    data_hists.append(hist)

for val_data, val_label, train_data, train_label in get_folds(data_hists, labels):
    for x in train_data:
	svm.train(train_data, train_label, params=svm_params)

"""
    hist = np.zeros(kmeans)
    for i in classified_points:
	hist[i[0]] += 1
"""

