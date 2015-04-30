import cv2
import numpy as np
import random
import operator
from os import listdir, walk
from os.path import isdir, isfile
from util import Spinner
from skimage import feature, color
from PIL import Image

import pdb

# TODO does not handle file nested more than 1 level deep
def get_filenames(directory):
    classes = {}
    count = 0
    for d in (x for x in listdir(directory) if not x.startswith('.')):
	classes[directory + d] = count
	count += 1
    for fldr in (f for f in listdir(directory) if isdir(directory + f)):
	for image_file in (f for f in listdir(directory + fldr) if f.split('.')[-1] == 'jpeg'):
	    full_path = './' + directory + fldr + '/' + image_file
	    # print full_path
	    yield (full_path, classes[directory + fldr])

def get_buckets(data, tup=False, pct=0.75, shuffle=False, kmeans=100, crit=(cv2.TERM_CRITERIA_EPS, 30, 0.1)):
    if pct <= 0 or pct > 1.0:
	print "Error: pct must be on interval (0, 1]"
	return None
    if kmeans < 2:
	print "Error: kmeans must be greater than 1"
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

def get_histograms(data, means, load=True, load_file='hist_save.npy',
	save=True, save_file='hist_save.npy'):
    data_hists = load_data(load_file)
    if len(data_hists) == 0:
	print "Building histograms"
	spinner = Spinner()
	for i, item in enumerate(data): # (img, label, SIFT descriptors) triple
	    hist = np.zeros(means.shape[0])
	    for desc in item[0]: # for each descriptor set each descriptor has 128 values)
		percent = int((float(i) / float(len(data))) * 100)
		spinner.spin(percent)
		dists = []
		for mindex, mean in enumerate(means):
		    dists.append((mindex, np.linalg.norm(mean - desc)))
		dists = sorted(dists, key=lambda entry: entry[1])
		hist[dists[0][0]] += 1
	    data_hists.append(np.array(hist))
	np.save(save_file, np.asarray(data_hists))
    return data_hists

def get_image_data(directory, selector='sift', **kwargs):

    load = kwargs.pop('load', True)
    save = kwargs.pop('save', True)
    load_file = kwargs.pop('load_file', selector + '_data_save.npy')
    save_file = kwargs.pop('save_file', selector + '_data_save.npy')
    scalex = kwargs.pop('scalex', 1.0)
    scaley = kwargs.pop('scaley', 1.0)
    selector = selector.lower()
    hog_ppc = kwargs.pop('hog_ppc', (8, 8))
    hog_cpb = kwargs.pop('hog_cpb', (3, 3))

    if selector not in ['sift', 'surf', 'hog']:
	print "Error: selector {} not supported".format(selector)
	exit(-1)

    if selector in ['sift', 'surf']:
	try:
	    inst = kwargs.pop('inst')
	except KeyError:
	    print """Error: SIFT/SURF given as feature selection method but no instance to
		a SIFT or SURF object was given"""
	    exit(-1)

    done = 0
    cant_calc = 0
    spinner = Spinner();

    data = load_data(load_file) if load else []

    if len(data) == 0:
	print "Loading images from {}".format(directory)

	dims = [Image.open(filepath[0]).size for filepath in get_filenames(directory)]

	mwidth, mheight = np.mean([dim[0] for dim in dims]), np.mean([dim[1] for dim in dims])
	mheight = mheight - mheight % 10
	mwidth, mheight = int(mwidth), int(mheight)
	size = kwargs.pop('size', (mwidth, mheight))

	print "Scaling images to {} and by a factor of {}x, {}y".format(size, scalex, scaley)
	print "Selecting {} features".format(selector)
	if selector == 'hog':
	    print "pixels per cell: {}, cells per block: {}".format(hog_ppc, hog_cpb)
	for filepath, label in get_filenames(directory):
	    spinner.spin(done, pct=False)
	    filename = filepath.split('/')[-1]
	    img = cv2.imread(filepath)
	    img = cv2.resize(img, size)
	    if scalex != 1.0 or scaley != 1.0:
		img = cv2.resize(img, (0, 0), fx=scalex, fy=scaley)
	    if selector == 'sift' or selector == 'surf':
		descriptors = get_descriptors(img, inst)
		if descriptors is not None:
		    data.append((descriptors, label))
		else:
		    cant_calc += 1
	    elif selector == 'hog':
		descriptors = feature.hog(color.rgb2gray(img),
			pixels_per_cell=hog_ppc, cells_per_block=hog_cpb)
		data.append((descriptors, label))
	    elif selector == 'log':
		print "LoG feature detection is not yet implemented"
		exit(-1)
	    done += 1
	if save:
	    print "Saving {} features to {}".format(len(data), save_file)
	    np.save(save_file, data)

    if cant_calc > 0:
	print "Unable to calculate features for {} images".format(cant_calc)
    return data

def load_data(filename):
    try:
	data = np.load(filename)
	print "Loaded {} items from {}".format(len(data), filename)
    except IOError:
	print "Unable to load data from {}".format(filename)
	data = []
    return data

def get_descriptors(img, inst, gray=False):
    if gray:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = inst.detectAndCompute(img, None)
    return des

