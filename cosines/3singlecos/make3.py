import cv2
import numpy as np
from PIL import Image

width= 1920
height= 1080
periods= 1


def makeimage(w, h, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/3.0 + wvcount*float(i)/float(w))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    ima = np.transpose(ima)
    cv2.imwrite(str(phi + 1) + 'cos.jpg', ima)


makeimage(width, height, periods, -1)
makeimage(width, height, periods, 0)
makeimage(width, height, periods, 1)

