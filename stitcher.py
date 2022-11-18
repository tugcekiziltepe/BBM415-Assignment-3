import cv2
import numpy as np
from numpy.core.fromnumeric import shape

def stitcher(image1, image2, image3):
    """This function provides stitching between 3 images.
    1. It finds keypoints and homography matrix of image2 and image1
    2. It warps perspective

    Args:
        image1 (numpy.ndarray): image on the middle
        image2 (numpy.ndarray): image on the right
        image3 (numpy.ndarray): image on the left

    """
    
    # Make bordder so that pasting other images will fit the resultant image.
    image1 = cv2.copyMakeBorder(image1,400,400,700,700, cv2.BORDER_CONSTANT)
    
    # calculate homography  for image2 to image1.
    (homography1, keypoints1, keypoints2, mask) = transform(image2,image1)
    
    # warp perspective using image2 and homogpraphy matrix
    output1 = cv2.warpPerspective(image2, homography1, (image1.shape[1],image1.shape[0]), dst=image1.copy(),borderMode=cv2.BORDER_TRANSPARENT)

    # calculate homography transformation for image3 to image1.
    (homography2, keypoints3, keypoints4, mask1) = transform(image3,image1)
    
    # warp perspective using image3 and homogpraphy matrix
    output2 = cv2.warpPerspective(image3, homography2, (image1.shape[1],image1.shape[0]), dst=image1.copy(),borderMode=cv2.BORDER_TRANSPARENT)

    # blend two outputs of warpPerspective() function
    # laplacian blending takes half of first image and the other half of second image
    output_image = laplacian_blending(output2, output1, 10) # call laplacian_blending fuction with num_levels = 4
    cv2.imwrite("output1.jpg", output1)
    cv2.imwrite("output2.jpg", output2)

    if not isinstance(output_image, type(None)):
        print("Image is saved.")
        cv2.imwrite("result.jpg", output_image) # write results
    else: 
        print("Stitching cannot be done.")


def draw_match(image1, image2, keypoints1, keypoints2, matches, matchesMask):
    """This function draws matches using cv2.drawMatchesKnn().
    Then writes this image to matches.jpg.

    Args:
        image1 (numpy.ndarray): first image
        image2 (numpy.ndarray): second image
        keypoints1 (numpy.ndarray): keypoints of first image
        keypoints2 (numpy.ndarray): keypoints of second image
        matches (numpy.ndarray): matches between keypoints of images
        matchesMask (numpy.ndarray): 	Mask determining which matches are drawn. 
    """
    matches_image = cv2.drawMatchesKnn(image1,keypoints1,image2,keypoints2, matches,None, matchesMask = matchesMask)
    cv2.imwrite("matches.jpg", matches_image)


def transform(image1, image2):
    """This function gets keypoints and descriptors of two image.
    Match them using FlannBasedMatcher
    Then return homography matrix of these two keypoints using RANSAC Algorithm.

    Args:
        image1 (numpy.ndarray): first image
        image2 (numpy.ndarray): second image
    """
    # create sift object
    sift = cv2.xfeatures2d.SIFT_create()

    # get the keypoints and descriptors of image1 and image2 using SIFT
    keypoints1, descriptor1 = sift.detectAndCompute(image1,None)
    keypoints2, descriptor2 = sift.detectAndCompute(image2,None)

    matcher = cv2.BFMatcher() # create marcher
    initial_matches = matcher.knnMatch(descriptor1,descriptor2,k=2) # get initial_maches
    
    matchesMask = [[0,0] for i in range(len(initial_matches))] # it is used to draw matches
    matches = [] # initialize matches array
    ratio = 0.75 # Lowe's ratio about distance
    for i, match in enumerate(initial_matches):
        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            matches.append((match[0].trainIdx, match[0].queryIdx))
            matchesMask[i]=[1,0]

    if len(matches) > 4:
        # draw matches
        draw_match(image1, image2, keypoints1, keypoints2, initial_matches, matchesMask)

        # convert KeyPoint object to numpy.ndarray
        keypoints1 = np.float32([key_point.pt for key_point in keypoints1])
        keypoints2 = np.float32([key_point.pt for key_point in keypoints2])
        keypoints1 = np.float32([keypoints1[i] for (_, i) in matches])
        keypoints2 = np.float32([keypoints2[i] for (i, _) in matches])

        # calculate homograph matrix using RANSAC Algorithm
        H, mask = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC, 5)

        # reshape keypoints
        keypoints1 = np.float32(keypoints1).reshape(-1,1,2)
        keypoints2 = np.float32(keypoints2).reshape(-1,1,2)
        
        return (H, keypoints1, keypoints2, mask)
    else:
        print("Not enough matches are found.")
        return


def blur_downsample(image):
    """Apply gaussian blur than downsample image

    Args:
        image (numpy.ndarray)

    Returns:
        numpy.ndarray: downsampled image
    """
    image = cv2.GaussianBlur(image, (5,5), 5) # apply gaussian blur
    ratio = 0.5 # half the size of image
    return cv2.resize(image, (0,0),fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)


def upsample(image,shape):
    """upsample image

    Args:
        image (numpy.ndarray): [description]

    Returns:
        numpy.ndarray: upsampled image
    """
    ratio = 2 # how much it will upsampled
    return cv2.resize(image,  shape,fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)


def laplacian_blending(image1, image2, num_levels):
    """ laplacian blending applied with half black half white mask.

    Args:
        image1 (numpy.ndarray): first image
        image2 (numpy.ndarray): second image

    Returns:
        output: output image as numpy.ndarray
    """
    image1_copy = image1.copy() # copy image1
    image2_copy = image2.copy() # copy image2
 
    gp_image1 = [image1_copy]    # gaussian pyramid of image1
    gp_image2 = [image2_copy]    # gaussian pyramid of image2

    for i in range(num_levels):
        image1_copy = blur_downsample(image1_copy)  # down sample image
        gp_image1.append(image1_copy.astype('float32'))

        image2_copy = blur_downsample(image2_copy)
        gp_image2.append(image2_copy.astype('float32'))

    # Generating Laplacin pyramids for both images
    lp_image1 = [gp_image1[num_levels]]
    lp_image2 = [gp_image2[num_levels]]

    for i in range(num_levels,0,-1):
        upsampled_image1 = upsample(gp_image1[i], (gp_image1[i-1].shape[1],gp_image1[i-1].shape[0])) # upsample image1
        # subract upper level gaussian pyramid and get laplacian pyramid
        # append result to laplacian pyramid array
        lp_image1.append(np.subtract(gp_image1[i-1], upsampled_image1))

        upsampled_image2 = upsample(gp_image2[i],  (gp_image2[i-1].shape[1],gp_image2[i-1].shape[0])) # upsample image2
        # subract upper level gaussian pyramid and get laplacian pyramid
        # append result to laplacian pyramid array
        lp_image2.append(np.subtract(gp_image2[i-1], upsampled_image2))

    LS = []
    for l_image1,l_image2 in zip(lp_image1,lp_image2):
        rows, cols, dims = l_image1.shape

        mask1 = np.zeros(l_image1.shape) # first mask is same shape with
        mask2 = np.zeros(l_image2.shape)

        mask1[:, 0:int(cols/ 2)] = 1 # first half of first image
        mask2[:, int(cols / 2):] = 1 # second half of second image
        
        LS.append(l_image1 * mask1 + l_image2 * mask2)
    
    output = LS[0] # largest image
    for i in range(1,num_levels+1):
        output = upsample(output, (LS[i].shape[1], LS[i].shape[0]))   # upsample image
        output = np.add(output, LS[i])
    
    return output
