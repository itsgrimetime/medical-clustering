import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
from sklearn import svm
import pylab as pl
from detect import *

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

kmeans = 200
data_dir = "./../Training Data/"
# selector = cv2.SIFT()
sift = cv2.SIFT()
# surf.extended = True
# svm = cv2.SVM()
# save_file = 'sift_save.npy'
save_file = 'sift_save.npy'
data_save_file = 'data' + save_file
means_save_file = 'means' + save_file
hist_save_file = 'hist' + save_file
crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)

ppc = (10, 10)
cpb = (3, 3)

test_labels = []
for line in open('../test_labels.txt'):
    test_labels.append([int(i.strip()) for i in line.split(',')])

test_data = get_image_data('./../Test Data/', selector='sift', inst=sift, **{'load': True, 'save': True, 'scalex': 0.25, 'scaley': 0.25, 'save_file': 'testsift.npy', 'load_file': 'testsift.npy'})


data = get_image_data(data_dir, selector='sift', inst=sift, **{'scalex': 0.25, 'scaley': 0.25, 'load': True, 'save': True})

means = load_data(means_save_file)
if len(means) == 0:
    print "Calculating bucket of words"
    means = get_buckets(data, tup=True, shuffle=True, kmeans=200)
    np.save(means_save_file, means)

labels = np.array([d[1] for d in data], dtype=np.float32)

data = np.array(get_histograms(data, means), dtype=np.float32)
test_data = np.array(get_histograms(test_data, means, load_file='test_hist_save.npy', save_file='test_hist_save.npy'), dtype=np.float32)

"""
svm_params = dict(kernel_type = cv2.SVM_RBF,
	svm_type = cv2.SVM_C_SVC,
	C=c, gamma=gamma)
"""

training_err = []
val_err = []
test_err = []
# clf = svm.SVC(kernel='linear', probability=True)
clf = svm.SVC(C=1000.0, kernel='rbf', gamma=0.0001, probability=True)
# clf = svm.LinearSVC()

# data = np.array([np.asarray(d[0]) for d in data], dtype=np.float32)
# test_data = np.array([np.asarray(d[0]) for d in test_data], dtype=np.float32)

spinner = Spinner();
print "Beginning Training of SVM and classifying"
for i in range(10):
    for val_data, val_label, train_data, train_label in get_folds(data, labels, shuffle=True):
	spinner.spin(shownum=False)
	#svm.train(train_data, train_label, params=svm_params)
	clf.fit(train_data, train_label)
	#train_result = svm.predict_all(train_data)
	train_result = clf.predict(train_data)
	# train_dec = clf.decision_function(train_data)
	mask = (train_result.flatten() == train_label)
	correct = np.count_nonzero(mask)
	training_err.append(correct * 100.0 / train_result.size)
	test_result = clf.predict(test_data)
	test_probs = clf.predict_proba(test_data)

	test_correct = 0.0
	for l, probs in enumerate(test_probs):
	    for class_guess in [i for i, prob, in enumerate(probs) if prob >= 0.25]:
		if class_guess in test_labels[l]:
		    test_correct += 1.0 / len(test_labels[l])

	test_err.append(test_correct * 100.0 / len(test_labels))
	#val_result = svm.predict_all(val_data)
	val_result = clf.predict(val_data)
	mask = (val_result.flatten() == val_label)
	correct = np.count_nonzero(mask)
	val_err.append(correct * 100.0 / val_result.size)
print "training error: {}".format(100.0 - np.mean(training_err))
print "validation error: {}".format(100.0 - np.mean(val_err))
print "testing error: {}".format(100.0 - np.mean(test_err))
