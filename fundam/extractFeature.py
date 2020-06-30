import cv2
import numpy as np

# Extract features and matching features 
def getPointPair(gray_image1, gray_image2, number_of_points):

    sift = cv2.xfeatures2d.SIFT_create(number_of_points, contrastThreshold = 0.04, sigma = 1.6)
    kp1, des1 = sift.detectAndCompute(gray_image1,None)
    kp2, des2 = sift.detectAndCompute(gray_image2,None)

    # Matching using BFmatcher or FLANN based matcher
    bf = cv2.BFMatcher()
    # flann = cv2.FlannBasedMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good_matches.append([m])

    # Get points pairs from good matches
    local_pairs = []
    des_pair = []
    for m in good_matches:
        p1 = kp1[m[0].queryIdx].pt
        p2 = kp2[m[0].trainIdx].pt
        des_pair.append([des1[m[0].queryIdx], des2[m[0].trainIdx]])
        local_pairs.append([p1, p2])
    des_pair = np.array(des_pair)
    return local_pairs, des_pair[:,-1]
