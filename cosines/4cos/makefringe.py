import cv2
import numpy as np
from PIL import Image

width = 1920
height = 1080
periods = 6


def make_image(w, h, wv_count, phi):
    ima = np.zeros((w, h))
    ima_line = np.ones(w)
    for i in range(w):
        ima_line[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/4.0 - wv_count*float(i)/float(w))))
    print(ima_line)
    for j in range(h):
        ima[:, j] = ima_line
    ima = np.transpose(ima)
    print(ima.shape)
    cv2.imwrite(str(phi-1) + '_cos.jpg', ima)


make_image(width, height, periods, 5)
make_image(width, height, periods, 6)
make_image(width, height, periods, 7)
make_image(width, height, periods, 8)
