from timeit import default_timer as timer
import tkinter
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
from math import sqrt


MAX_FEATURES = 50000
GOOD_MATCH_PERCENT = 0.25
MIN_MATCHES = 10
camindex = 2
ransacReprojThreshold = 25.0
im1 = cv2.imread('template.jpg')
im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)


def findmatch(im2):

   # try:
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    templatekps = cv2.drawKeypoints(im1, keypoints1, None, color=(0, 255, 0), flags=0)
    cv2.imwrite('templatekeys.jpg', templatekps)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    if(len(matches)<MIN_MATCHES):
        raise Exception('Not enough matches')
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold = ransacReprojThreshold)
    height, width, channels = im1.shape
    pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,h)
    #cen = quadcentroid(dst)
    #qsz = quadsize(dst)
    cv2.polylines(im2,[np.int32(dst)],True,(255, 0, 255),3, cv2.LINE_AA)
    iMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite('matches.jpg', iMatches)
    #display = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(iMatches, cv2.COLOR_BGR2RGB)))
    return #cen, qsz#, display

    #except:
        #display = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)))
        #return (0,0), 1#, display
    
    
img2 = cv2.imread('scene.jpg')
findmatch(img2)
