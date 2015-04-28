import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.io import imread
from skimage.transform import resize
from os import listdir
from os.path import isdir

data_dir = "Training Data/"

# image = imread('foot.jpeg', flatten=True)

pp_images = []
folders = listdir('./' + data_dir)

print folders

i = 0
for folder in folders:
    if isdir('./' + data_dir + folder):
	print folder
	for image_file in listdir('./' + data_dir + folder):
	    if image_file.split('.')[-1] == 'jpeg':
		print "\t" + image_file
		full_path = './' + data_dir + folder + '/' + image_file
		pp_images.append(resize(imread(full_path, flatten=True), (400, 400)))
		i += 1

for image in pp_images:
    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh

    block_size = 40
    binary_adaptive = threshold_adaptive(image, block_size, offset=10)

    fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
    ax0, ax1, ax2 = axes
    plt.gray()

    ax0.imshow(image)
    ax0.set_title('Image')

    ax1.imshow(binary_global)
    ax1.set_title('Global thresholding')

    ax2.imshow(binary_adaptive)
    ax2.set_title('Adaptive thresholding')

    for ax in axes:
	    ax.axis('off')
	    plt.show()

