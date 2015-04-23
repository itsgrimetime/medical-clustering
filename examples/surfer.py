import cv2
import numpy as np
from operator import attrgetter

img = cv2.imread('foot.jpeg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.SURF()
kp, des = surf.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, sorted(kp, key=attrgetter('size'))[::-1][0:20], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('surf-keypoints.jpg', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
