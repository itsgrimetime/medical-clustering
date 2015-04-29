== Run #1 ==

Kernelized SVM (libsvm C-Support Vector Classifier) (sklearn) with following parameters:
    - C = 1.0
    - RBF(gamma = 0.1)

Data:
- Unscaled, unprocessed images (not including images in True-Negatives directory)

Feature Selection
SIFT - criteria: (CV2.TERM_CRITERIA_EPS, 30, 0.1)

A random selection of 75% of the SIFT descriptors of the entire training set were
then grouped bag-of-words style using k-means clustering (k = 200).

The training set SIFT descriptors were then fit into histograms, based on 2-norm
distance from the previously calculated means.

Using k-fold CV (k = 5), and without random shuffling of data, the following error
rates were observed:

Training Error: 0.3125%
Validation Error: 98.0%

With random shuffling of data before training the SVM, the following error rates
were recorded:

Training Error: 0.25%
Validation Error: 75.75%

Testing images dont have labels, so I cant get testing error right away :(

=== Narrowing down C and Gamma values for C-SVM and RBF ===

gamma values between 1e-05 and 1000.0 (logarithmic) were used, as well as
C values of 0.01 to 100000000.0 (logarithmic) were combined using the same 5-Fold CV.

with the following training and validation errors:

<figures>

looks like C values between 1e3 and 1e8 with a gamma of 1e-4 give the best tradeoff
between low training and low validation errors. well keep playing with them within
these ranges.

With a C-value of 1000.0 and a gamma of 1e-4:

Training Error: 0.4625%
Validation Error: 36.25%

sklearn predict_proba function returns a vector of the probabilities that the data
point lies in each class. In order to get multi-class membership Im just going to
add membership to the class if the probability for that class is over 25%. I might
change that to 33%, as it might be reasonable to assume that a datapoint wont ever
be in more than 3 classes (I havent seen any full-body xrays yet)

This is weird though, because the training data doesnt have multi-class labels even
though the objective asks for multi-class membership predictions :(

I could manually do it though...

=== Other Features ===

Looking at the rendering of the SIFT features, its hard to tell intuitively see
what they are really describing. LoG on the otherhand, looks as though the points
accurately describe bones and other features of the X-Ray.

- try LoG
- try SURF

=== Other Classification Models ===

- try kNN

=== Try removing keypoints that are in those white boxes 8^) ===
