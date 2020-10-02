import cv2
import numpy as np
from PIL import Image

width = 4096
height = 2732
periods = 80


def makeimage(w, h, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/4.0 + wvcount*float(i)/float(w))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    ima = np.transpose(ima)
    cv2.imwrite('/home/samir/dblive/cosines/singleshotcosine/'+ str(phi + 1) + '_cos.jpg', ima)


makeimage(width, height, periods, -1)
makeimage(width, height, periods, 0)
makeimage(width, height, periods, 1)
makeimage(width, height, periods, 2)

