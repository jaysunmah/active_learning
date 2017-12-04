'''
Attempt to process image into simpler filter
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def threshold(x):
    if x <= 20:
        return 0
    return x

def process(path):
    image = cv2.imread(path)

    # cv2.imshow('original', image)

    image = np.float32(image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 99.0)
    K = 32
    ret,label,center=cv2.kmeans(image,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    gray_image = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    colors = set()
    window_size = 5
    print(gray_image.shape)
    for r in range(len(gray_image)):
        for c in range(len(gray_image[0])):
            if gray_image[r][c] not in colors:
                colors.add(gray_image[r][c])
    print(len(colors))

    # kernel_size = 4
    # kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size * kernel_size)
    # dst = cv2.filter2D(gray_image,-1,kernel)

    # plt.hist(gray_image.ravel(),256,[0,256]); plt.show()

    # gray_image = threshold(gray_image)



    cv2.imshow('res2',gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process("data/all/good/bottle0_85.jpg")
