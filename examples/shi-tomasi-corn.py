"""
Source: http://opencv-python-tutroals.readthedocs.org/
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('foot.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x, y), 8, 255, -1)

cv2.imwrite('shi-tomasi-corners.jpeg', cv2.resize(img, (0, 0), fx=0.25, fy=0.25))
