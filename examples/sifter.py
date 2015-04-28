import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random

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

toolbar_width = 80
kmeans = 100
data_dir = "./../Training Data/"
sift = cv2.SIFT()
svm = cv2.SVM()
save_file = 'sift_save'
desc_save_file = 'desc' + save_file


# setup toolbar
"""
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
"""

# TODO make sure this is right place to be doing this and finish it
# for val_data, val_label, train_data, train_label in get_folds

loaded_desc = False
try:
    all_descriptors = np.load(desc_save_file)
    print "Loaded descriptors from {}".format(desc_save_file)
    loaded_desc = True
except IOError:
    all_descriptors = np.empty((0, 128), dtype=np.float32)

num_files = 0
for filename, label in get_filenames(data_dir):
    print "filename: {}, label: {}".format(filename, label)
    img = cv2.imread(filename)
    all_descriptors = np.vstack((all_descriptors, get_descriptors(img, sift)))
    num_files += 1

print "{} image descriptors".format(num_files)

"""
chunk_size = len(pp_images) / (toolbar_width - 1)
for i in xrange(toolbar_width):
    sys.stdout.write("-")
    sys.stdout.flush()
sys.stdout.write("\n")
"""
if not loaded:
    print "Saving {} descriptors to {}".format(len(all_descriptors), desc_save_file)
    np.save(desc_save_file, all_descriptors)

print("Total SIFTs: {}".format(count))

# temp, classified_points, means = cv2.kmeans(np.asarray(all_descriptors), K=2, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, kmeans), attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
print("Starting K-Means")
temp, classified_points, means = cv2.kmeans(all_descriptors, K=kmeans, bestLabels=None, criteria=crit, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
hist = np.zeros(kmeans)
for i in classified_points:
    hist[i[0]] += 1
# n, bins, patches = plt.hist(hist, kmeans, facecolor='green', alpha=0.5)
# plt.show()

svm_params = dict( kernel_type = cv2.SVM_RBF,
	svm_type = cv2.SVM_C_SVC,
	C=1.0, gamma=0.1)

svm.train(train_data, labels, params=svm_params)

