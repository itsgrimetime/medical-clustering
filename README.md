# ImageCLEF 2015 - Medical Clustering

## Task

Grouping of digital x-ray images into four clusters: head-neck, upper-limb, body, and lower-lib.
Further sub-clustering of initial groups into more specific sub-groups (such as specific bones)
should be kept as a secondary goal.

## Data

500 Digital x-rays, collected over a year, categorized by their general anatomical location:
head-neck, upper-limb, body, and lower-limb. Pictures vary in resolution from 1600x1600 - 1600x3500.

### Proposed feature extraction method(s)
- shapes
- vectors that describe lengths of bone structures
	- relative to other bones in picture
	- or just absolute in picture
- maybe other ones if I think of them
- get blobs using skimage.feature (blob_dog, other blob_* functions)
- figure out a way to maybe get edges
- figure out a way to get vectors that describe the image

### Proposed training model(s)
- look at other papers that have done similar things and maybe try to add something new
- who knows maybe kNN??????????

### Preprocessing
can probably get away with scaling images down a lot

