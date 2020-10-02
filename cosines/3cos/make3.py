import cv2
import numpy as np
from PIL import Image
# from scan.pylib.gamma import *

width = 400
height = 400
periods = 1
hf_periods = 20


def makeimage(w, h, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    raw_inp = np.ones(w)
    for i in range(w):
        raw_inp[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(1.0*float(phi)/3.0 + wvcount*float(i)/float(w))))
        # imaline[i] = np.polyval(gamma_correct, raw_inp[i])
        imaline[i] = raw_inp[i]
    for j in range(h):
        ima[:, j] = imaline
    ima = np.transpose(ima)
    cv2.imwrite(str(phi + 1) + '_cos.jpg', ima)


def maketexture(w, h, value):
    ima = np.full((w,h), value)
    ima = np.transpose(ima)
    cv2.imwrite('texture.png', ima)

# file = '/home/samir/db2/scan/static/scan_folder/gamma_im_folder/image1.png'
# gamma_correct = compensate_gamma(file)

makeimage(width, height, hf_periods, -1)
makeimage(width, height, hf_periods, 0)
makeimage(width, height, hf_periods, 1)
makeimage(width, height, periods, 5)
makeimage(width, height, periods, 6)
makeimage(width, height, periods, 7)
maketexture(width, height, 100)
