# BBM415-Assignment-3

In this assignment, we will generate panoramic images by stitching 3 images. Panoramic images will be created by using registering, warping, resampling and blending algorithms.

## Algorithm
1. Read all of the images in an order
2. Detect keypoints and extract SIFT from two input images
3. Match the descriptors of the images
4. Use RANSAC algorithm to estimate a homography matrix using matched
feature vectors
5. Using the homography matrix obtained, apply wrapping
6. Lastly, apply image blending function (pyramid blending)
