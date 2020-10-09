# This module plots a row in 2 different files for comparing fringe and wral patterns between different files


import matplotlib.pyplot as plt
import cv2
import numpy as np

H=160
W= 160


def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print( img.shape )
    return(img)


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img



def makemonohigh(folder):
    high = folder + '/blenderimage0.png'
    colorhigh = cv2.imread(high, 1)
    monohigh = make_grayscale(colorhigh)
    cv2.imwrite(folder+'monohigh.png', monohigh)
    return

folder1 = '/home/samir/Desktop/blender/pycode/160planes/render2'
folder2 = '/home/samir/Desktop/blender/pycode/160planes/render12'

monohigh = np.zeros((H, W), dtype=np.float64)

high = folder2 + '/cnnwrap1.png'
colorhigh = cv2.imread(high, 1)
colorhigh = resize(colorhigh, W, H)
monohigh1 = make_grayscale(colorhigh)

high = folder1 + '/cnnwrap1.png'
colorhigh = cv2.imread(high, 1)
colorhigh = resize(colorhigh, W, H)
monohigh2 = make_grayscale(colorhigh)


x = range(160)

for i in range(0,160,20):
    plt.plot(x, monohigh1[:,i],
    x, monohigh2[:,i])
    plt.ylabel(str(i)) 
    plt.show()