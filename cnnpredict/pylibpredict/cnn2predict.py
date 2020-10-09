# Cnn2 as per paper, processing to generate cos and sin Nom and denom
# 4newcnn2a for a low freq cos
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers
from keras.models import Sequential, Model, Input
from keras import optimizers
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Lambda
from keras.layers import Conv2D, MaxPooling2D, Add, UpSampling2D, concatenate, Concatenate, AveragePooling2D
from keras.utils import plot_model
import time
import nnwrap as nnw


PI = np.pi
INPUTHIGHT = 170
INPUTWIDTH = 170
rheight = 170
rwidth = 170
high_freq = 13
low_freq = 1

def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)number_of_epochs = 5
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def load_high_modelold():
    model = keras.models.load_model('/home/samir/dblive/cnnpredict/cnnmodels/cnn2a-bmodel-shd-1npy-465-40.h5')
    return(model)



def load_low_modelold():
    model = keras.models.load_model('/home/samir/dblive/cnnpredict/cnnmodels/cnn2a-bmodel-shd-4npy-465-449-200.h5')
    return(model)




def nnswat_wrap(nom, denom):
    wrap = np.zeros((rheight, rwidth), dtype=np.float)
    im_wrap = np.zeros((rheight, rwidth), dtype=np.float)
    greynom = np.load(nom)
    greynom = normalize_image(greynom)
    greydenom = np.load(denom)
    greydenom = normalize_image(greydenom)
    for i in range(rheight):
        for j in range(rwidth):
            wrap[i, j] = np.arctan2(greynom[i, j], greydenom[i, j])
            if wrap[i, j] < 0:
                if greynom[i, j] < 0:
                    wrap[i, j] += 2*np.pi
                else:
                    wrap[i, j] += 1 * np.pi
            im_wrap[i, j] = 128/np.pi * wrap[i, j]
    # wrap = cv2.GaussianBlur(wrap, (3, 3), 0)
    # im_wrap = cv2.GaussianBlur(im_wrap, (3, 3), 0)
    return(wrap, im_wrap)



def save1swat(nom, denom):
    folder = '/home/samir/dblive/scan/static/scan_image_folder/'
    np.save(folder+ 'nomhigh.npy', nom, allow_pickle=False)
    np.save(folder+ 'denomhigh.npy', denom, allow_pickle=False)


def save4swat(nom, denom):
    folder = '/home/samir/dblive/scan/static/scan_image_folder/'
    np.save(folder+ 'nomlow.npy', nom, allow_pickle=False)
    np.save(folder+ 'denomlow.npy', denom, allow_pickle=False)


def save1nnwrap(folder):
    nnnom = folder+ 'nomhigh.npy'
    nndenom = folder+ 'denomhigh.npy'
    nnnpywrap, nnimwrap = nnswat_wrap(nnnom, nndenom)
    cv2.imwrite(folder+ 'nnhighwrap.png', nnimwrap)
    np.save( folder+ 'nnhighwrap.npy', nnnpywrap, allow_pickle=False)


def save4nnwrap(folder):
    nnnom = folder+ 'nomlow.npy'
    nndenom = folder+ 'denomlow.npy'
    nnnpywrap, nnimwrap = nnswat_wrap(nnnom, nndenom)
    cv2.imwrite(folder+ 'nnlowwrap.png', nnimwrap)
    np.save( folder+ 'nnlowwrap.npy', nnnpywrap, allow_pickle=False)


def makemonohigh(folder):
    high = folder + 'high.png'
    colorhigh = cv2.imread(high, 1)
    monohigh = make_grayscale(colorhigh)
    cv2.imwrite(folder+'monohigh.png', monohigh)
    return


def mask(folder):
    color = folder + 'color.png'
    img1 = np.zeros((INPUTHIGHT, INPUTWIDTH), dtype=np.float)
    img1 = cv2.imread(color, 1).astype(np.float32)
    gray = make_grayscale(img1)


    black = folder + 'black.png'
    img2 = np.zeros((INPUTHIGHT, INPUTWIDTH), dtype=np.float)
    img2 = cv2.imread(black, 0).astype(np.float32)
    diff1 = np.subtract(gray, .5*img2)
    mask =  np.zeros((INPUTHIGHT, INPUTWIDTH), dtype=np.float)
    for i in range(INPUTHIGHT):
        for j in range(INPUTWIDTH):
            if (diff1[i,j]<50):
                mask[i,j]= True
    np.save( folder+ 'mask.npy', mask, allow_pickle=False)
    cv2.imwrite( folder+ 'mask.png', 128*mask)
    return(mask)




def nnHprocess(folder):
    high = folder + 'high.png'
    image = cv2.imread(high, 1).astype(np.float32)
    inp_1 = normalize_image255(image)
    inp_1 = make_grayscale(inp_1)

    color = folder + 'color.png'
    image = cv2.imread(color, 1).astype(np.float32)
    inp_2 = normalize_image255(image)
    inp_2 = make_grayscale(inp_2)

    start = time.time()
    predicted_img = high_model.predict(
        [np.array([np.expand_dims(inp_1, -1)]), np.array([np.expand_dims(inp_2, -1)])])
    predicted_img[0] = predicted_img[0].squeeze()
    predicted_img[1] = predicted_img[1].squeeze()
    end = time.time()

    mask = np.load(folder+'mask.npy')
    nom = np.multiply(np.logical_not(mask), predicted_img[0])
    denom = np.multiply(np.logical_not(mask), predicted_img[1])

    print('elapsed high:', end-start)
    cv2.imwrite( folder + 'nomhigh.png',255*nom)
    cv2.imwrite( folder + 'denomhigh.png' ,255*denom)
    save1swat(nom, denom)
    save1nnwrap(folder)
    return  #(predicted_img[0], predicted_img[1])

def nnLprocess(folder):
    low = folder + 'low.png'
    image = cv2.imread(low, 1).astype(np.float32)
    inp_1 = normalize_image255(image)
    inp_1 = make_grayscale(inp_1)

    color = folder + 'color.png'
    image = cv2.imread(color, 1).astype(np.float32)
    inp_2 = normalize_image255(image)
    inp_2 = make_grayscale(inp_2)

    start = time.time()
    predicted_img = low_model.predict(
        [np.array([np.expand_dims(inp_1, -1)]), np.array([np.expand_dims(inp_2, -1)])])
    predicted_img[0] = predicted_img[0].squeeze()
    predicted_img[1] = predicted_img[1].squeeze()
    end = time.time()

    mask = np.load(folder+'mask.npy')
    nom = np.multiply(np.logical_not(mask), predicted_img[0])
    denom = np.multiply(np.logical_not(mask), predicted_img[1])

    print('elapsed low:', end-start)
    cv2.imwrite(folder + 'nomlow.png', 255*nom)
    cv2.imwrite(folder + 'denomlow.png', 255*denom)
    save4swat(nom, denom)
    save4nnwrap(folder)
    return  #(predicted_img[0], predicted_img[1])







def unwrap_r(folder):
    filelow = folder + 'nnlowwrap.npy'
    filehigh = folder +  'nnhighwrap.npy'
    wraplow = np.zeros((rheight, rwidth), dtype=np.float64)
    wraphigh = np.zeros((rheight, rwidth), dtype=np.float64)
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float64)
    im_unwrap = np.zeros((rheight, rwidth), dtype=np.float64)
    wraplow = np.load(filelow)  # To be continued
    wraphigh = np.load(filehigh)
    print('highrange=', np.ptp(wraphigh), np.max(wraphigh), np.min(wraphigh) )
    print('lowrange=', np.ptp(wraplow), np.max(wraplow), np.min(wraplow) )
    # print('high:', wraphigh)
    # print('low:', wraplow)
    
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float64)
    kdata = np.zeros((rheight, rwidth), dtype=np.int64)
    # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
    # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
    for i in range(rheight):
        for j in range(rwidth):
            kdata[i, j] = round((high_freq/low_freq * (wraplow[i, j])- wraphigh[i, j])/(2*PI))
            # unwrapdata[i,j] = .1*(1.1*wraphigh[i, j]/np.max(wraphigh) +2*PI* kdata[i, j]/np.max(wraphigh))
            # unwrapdata[i,j] = 1*(1*wraphigh[i, j] +2*PI* kdata[i, j])
    unwrapdata = np.add(wraphigh, np.multiply(2*PI,kdata) )
    print('kdata:', np.ptp(np.multiply(1,kdata)))
    print('unwrap:', np.ptp(unwrapdata))
    # print("I'm in unwrap_r")
    print('kdata:', kdata[::40, ::40])
    wr_save = folder + 'unwrap.npy'
    np.save(wr_save, unwrapdata, allow_pickle=False)
    # print(wr_save)
    # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
    # unwrapdata = np.multiply(unwrapdata, 1.0)
    # unwrapdata = np.unwrap(np.transpose(unwrapdata))
    # unwrapdata = cv2.GaussianBlur(unwrapdata,(0,0),3,3)
    # unwrapdata = np.multiply(unwrapdata, 1.0)
    maxval = np.amax(unwrapdata)
    print('maxval:', maxval)
    # im_unwrap = 255*unwrapdata/ maxval# np.max(unwrapdata)*255)
    im_unwrap = 3*unwrapdata# np.max(unwrapdata)*255)
    # unwrapdata/np.max(unwrapdata)*255
    cv2.imwrite(folder + 'unwrap.png', im_unwrap)
    cv2.imwrite(folder + 'kdata.png', np.multiply(2*PI,kdata))

##########################################################################################################################


# folder = '/home/samir/dblive/scan/static/scan_image_folder/'
# mask(folder)
# makemonohigh(folder)
# high_model = load_high_model()
# low_model = load_low_model()
# nnHprocess(folder)
# nnLprocess(folder)
# unwrap_r(folder)
# print('done!')

