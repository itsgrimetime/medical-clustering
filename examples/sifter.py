import cv2
import numpy as np
from operator import attrgetter

num_keypoints = 50

img = cv2.imread('foot.jpeg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, sorted(kp, key=attrgetter('size'))[::-1][0:50], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints.jpg', cv2.resize(img, (0, 0), fx=0.25, fy=0.25))
