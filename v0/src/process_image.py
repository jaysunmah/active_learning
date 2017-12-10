'''
Attempt to process image into simpler filter
Code forked from existing project:
https://gist.github.com/Munawwar/0efcacfb43827ba3a6bac3356315c419
Modified to further suit our own project
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def getSobel (channel):

    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    sobel = np.hypot(sobelx, sobely)

    return sobel;

def findSignificantContours (img, sobel_8u):
    image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:
            cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant];

def threshold(x):
    return x if x == 0 else 255

# def segment(img):
def segment(path):
    img = cv2.imread(path)

    # blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
    blurred = cv2.blur(img,(15,15))

    # Edge operator
    sobel = np.max( np.array([ getSobel(blurred[:,:, 0]), getSobel(blurred[:,:, 1]), getSobel(blurred[:,:, 2]) ]), axis=0 )

    # Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    mean = np.mean(sobel)

    # Zero any values less than mean. This reduces a lot of noise.
    sobel[sobel <= mean] = 0;
    sobel[sobel > 255] = 255;
    cv2.imwrite('output/edge.png', sobel);

    sobel_8u = np.asarray(sobel, np.uint8)

    # Find contours
    significant = findSignificantContours(img, sobel_8u)

    # Mask
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Finally remove the background
    img[mask] = 0;

    kernel = np.ones((5,5),np.uint8)
    # we remove thin lines to get central blob
    img = cv2.erode(img,kernel,iterations = 10)
    ret,img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)

    h = img.shape[0]
    w = img.shape[1]
    img2 = np.zeros((h,w,3))
    img2[:,:,0] = img
    img2[:,:,1] = img
    img2[:,:,2] = img

    # return img2
    fname = path.split('/')[-1]
    cv2.imwrite('output/' + fname, img);

# segment("data/all/good/good_mug_391.jpg")
segment("data/all/bad/bad_mug_379.jpg")
