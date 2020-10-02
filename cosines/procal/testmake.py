import cv2
import numpy as np
from PIL import Image

width = 1920
height = 1080
periods = 24
v_periods = 18

def makeimage(w, h, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(w)
    for i in range(w):
        imaline[i] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/3.0 + wvcount*float(i)/float(w))))
    print(imaline)
    for j in range(h):
        ima[:, j] = imaline
    ima = np.transpose(ima)
    cv2.imwrite(str(phi + 1) + '_cos.jpg', ima)


def make_t_image(w, h, wvcount, phi):
    ima = np.zeros((w, h))
    imaline = np.ones(h)
    for j in range(h):
        imaline[j] = 255.0*(1.0/2.0 + 1.0/2.0*np.cos(2.0*np.pi*(float(phi)/3.0 + wvcount*float(j)/float(h))))
    print(imaline)
    for i in range(w):
        ima[i, :] = imaline
    ima = np.transpose(ima)
    cv2.imwrite(str(phi + 4) + '_cos.jpg', ima)


def make_h_line(w, h):
    ima = np.full((w, h), 10)
    imaline = np.full(w, 240)
    ima[:, round(h/2)] = imaline
    ima = np.transpose(ima)
    cv2.imwrite('7_cos.jpg', ima)


def make_v_line(w, h):
    ima = np.full((w, h), 10)
    imaline = np.full(h, 240)
    ima[round(w/2), :] = imaline
    ima = np.transpose(ima)
    cv2.imwrite('6_cos.jpg', ima)


def make_v_cv2_line(w,h):
    img = np.zeros((h, w, 3), np.uint8)
    # img = np.transpose(img)
    cv2.line(img, (int(round(w/2)), 0), (int(round(w/2)), h), (0, 255, 0), 3)
    cv2.imwrite('7_cos.jpg', img)


def make_h_cv2_line(w,h):
    img = np.zeros((h, w, 3), np.uint8)
    # img = np.transpose(img)
    cv2.line(img, (0, int(round(h/2))), (w, int(round(h/2))), (0, 255, 0), 3)
    cv2.imwrite('6_cos.jpg', img)


makeimage(width, height, periods, -1)
makeimage(width, height, periods, 0)
makeimage(width, height, periods, 1)
make_t_image(width, height, v_periods, -1)
make_t_image(width, height, v_periods, 0)
make_t_image(width, height, v_periods, 1)
# make_h_line(width, height)
# make_v_line(width, height)
# make_h_cv2_line(width, height)
# make_v_cv2_line(width, height)
