import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
import operator
from sklearn import svm

import pylab as pl

from os import listdir
from os.path import isdir

import time
import sys

def get_descriptors(img, sift, gray=False):
    if gray:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    return des

def get_folds(data, labels, k=5, shuffle=False):
    if len(data) != len(labels):
	print "Error: must have same number of data points and labels"
	yield None
    fold_size = len(data) / k

    if shuffle:
	shuf_index = range(len(data))
	random.shuffle(shuf_index)
	data = np.array([data[i] for i in shuf_index])
	labels = np.array([labels[i] for i in shuf_index])

    for chunk in range(k):
	val_data = np.array(data[chunk * fold_size : (chunk + 1) * fold_size])
	val_label = np.array(labels[chunk * fold_size : (chunk + 1) * fold_size])
	train_data = np.concatenate((data[0 : chunk * fold_size], data[(chunk + 1) * fold_size :]))
	train_label = np.concatenate((labels[0 : chunk * fold_size], labels[(chunk + 1) * fold_size :]))
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
    if pct <= 0 or pct > 1.0:
	print "Error: pct must be on interval (0, 1]"
	return None
    if kmeans < 2:
	print "Error: kmeans must be grearter than 1"
	return None
    index = range(len(data))
    if shuffle: random.shuffle(index)
    desc = np.empty((0, 128), dtype=np.float32)
    cut = int(round(len(data) * pct))
    for i in index[:cut]:
	if tup:
	    desc = np.vstack((desc, data[i][0]))
	else:
	    desc = np.vstack((desc, data[i]))

    temp, classified_points, means = cv2.kmeans(desc, K=kmeans, bestLabels=None,
	    criteria=crit, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
    return means

def load_data(filename):
    try:
	data = np.load(filename)
	print "Loaded {} items from {}".format(len(data), filename)
    except IOError:
	print "Unable to load data from {}".format(filename)
	data = []
    return data

def get_histograms(data, means):
    data_hists = load_data(hist_save_file)
    if len(data_hists) == 0:
	print "Building histograms"
	spinner = ['\\', '|', '/', '-']
	spincount = 0
	for i, item in enumerate(data): # (img, label, SIFT descriptors) triple
	    hist = np.zeros(means.shape[1])
	    for desc in item[0]: # for each descriptor set each descriptor has 128 values)
		percent = int((float(i) / float(len(data))) * 100)
		status_string = "{} ({}%)".format(spinner[spincount % len(spinner)], percent)
		sys.stdout.write(status_string)
		sys.stdout.flush()
		sys.stdout.write('\b' * len(status_string))

		dists = []
		for mindex, mean in enumerate(means):
		    dists.append((mindex, np.linalg.norm(mean - desc)))
		dists = sorted(dists, key=lambda entry: entry[1])
		hist[dists[0][0]] += 1
		spincount += 1
	    data_hists.append(np.array(hist))
	np.save(hist_save_file, np.asarray(data_hists))
    return data_hists


toolbar_width = 80
kmeans = 100
data_dir = "./../Training Data/"
sift = cv2.SIFT()
# svm = cv2.SVM()
save_file = 'sift_save.npy'
data_save_file = 'data' + save_file
means_save_file = 'means' + save_file
hist_save_file = 'hist' + save_file
crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)

# TODO you can refactor load_data to take a function to run when
# the data can't be loaded. Same with means down there
data = load_data(data_save_file)
if len(data) == 0:
    for filename, label in get_filenames(data_dir):
	print "filename: {}, label: {}".format(filename, label)
	img = cv2.imread(filename)
	data.append((get_descriptors(img, sift), label))
    print "Saving {} descriptors to {}".format(len(data), data_save_file)
    np.save(data_save_file, data)

# temp, classified_points, means = cv2.kmeans(np.asarray(all_descriptors), K=2, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, kmeans), attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
# def get_buckets(data, tup=False, pct=0.75, shuffle=False, kmeans=100):
# get our defining "bucket of words"

means = load_data(means_save_file)
if len(means) == 0:
    print "Calculating bucket of words"
    means = get_buckets(data, tup=True, shuffle=True)
    np.save(means_save_file, means)

labels = np.array([d[1] for d in data], dtype=np.float32)
data_hists = np.array(get_histograms(data, means), dtype=np.float32)

"""
svm_params = dict(kernel_type = cv2.SVM_RBF,
	svm_type = cv2.SVM_C_SVC,
	C=c, gamma=gamma)
"""

training_err = []
val_err = []
clf = svm.SVC(C=1000.0, kernel='rbf', gamma=0.0001, probability=False)

for i in range(10):
    for val_data, val_label, train_data, train_label in get_folds(data_hists, labels, shuffle=True):
	#svm.train(train_data, train_label, params=svm_params)
	clf.fit(train_data, train_label)
	#train_result = svm.predict_all(train_data)
	train_result = clf.predict(train_data)
	# train_probs = clf.predict_proba(train_data)
	# train_dec = clf.decision_function(train_data)
	mask = (train_result.flatten() == train_label)
	correct = np.count_nonzero(mask)
	training_err.append(correct * 100.0 / train_result.size)

	#val_result = svm.predict_all(val_data)
	val_result = clf.predict(val_data)
	mask = (val_result.flatten() == val_label)
	correct = np.count_nonzero(mask)
	val_err.append(correct * 100.0 / val_result.size)

print "training error: {}".format(100.0 - np.mean(training_err))
print "validation error: {}".format(100.0 - np.mean(val_err))


