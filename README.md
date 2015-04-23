# ImageCLEF 2015 - Medical Clustering

## Task

Grouping of digital x-ray images into four clusters: head-neck, upper-limb, body, and lower-lib.
Further sub-clustering of initial groups into more specific sub-groups (such as specific bones)
should be kept as a secondary goal.

## Description of Data

500 Digital x-rays, collected over a year, categorized by their general anatomical location:
head-neck, upper-limb, body, and lower-limb. Pictures vary in resolution from 1600x1600 - 1600x3500.

### Feature selection/extraction methods
- Shi-Tomasi Corner Detector (GoodFeaturesToDetect) (25 corners):

![Shi-Tomasi Corner Detection][shi-tomasi-corn]

- SIFT (50 largest keypoints):

![SIFT][sift]

- SURF
<!---![SURF][surf]-->
- Difference of Gaussians (`blob_dog()`)
- Laplacian of Gaussians (`blob_log()`)
- Other `blob_*()` methods
- Edge histogram

### Proposed training model(s)
- kNN
- Kernelized SVM or Logistic Regression

### Preprocessing Checkpoints
- [ ] Scale all images by certain factor
- [ ] Remove all white text boxes in images
- [ ] Crop all images to ensure only black background

### Todo
- [ ] Add images to README for examples of what different feature selection methods do

[shi-tomasi-corn]: https://github.com/magrimes/medical-clustering/blob/master/examples/shi-tomasi-corners.jpeg
[sift]: https://github.com/magrimes/medical-clustering/blob/master/examples/sift_keypoints.jpg
[surf]:
