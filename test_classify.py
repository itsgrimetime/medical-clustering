from os import listdir
from os.path import isdir

img_dir = "./Test Data/images/"

items = listdir(img_dir)

target = open('test_files.txt', 'w')

for item in items:
    target.write(item + '\n')


