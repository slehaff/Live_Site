from keras import layers
import numpy as np
import os
import cv2
# import boto3models/cnn0a-173-model50.h5
import pandas as pd
# from sagemaker import get_execution_role

import keras
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Add


IMAGECOUNT = 220
EPOCHS = 200


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print(img.shape)
    return(img)



# def get_my_file(infile):
#     s3 = boto3.resource('s3')
#     s3.Bucket('sagemakerdanbots1').download_file(
#         'data/' + infile, 'myfile.png')
#     return


def to_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        print(i)
        myfile = folder_path + str(i)+'.png'
        img = cv2.imread(myfile).astype(np.float32)
        img = normalize_image255(img)
        inp_img = make_grayscale(img)
        array.append(inp_img)
        print(i)
    return


def to_png_array(folder_path, filename, array, file_count):
    for i in range(file_count):
        myfile = folder_path + str(i)+'/'+ filename + '.png'
        img = cv2.imread(myfile).astype(np.float32)
        img = resize(img, 160, 160)
        maskfile = folder_path + str(i)+'/'+ 'mask' + '.npy'
        mask = np.load(maskfile)
        img = normalize_image255(img)
        inp_img = make_grayscale(img)
        img = np.logical_and(inp_img, np.logical_not(mask))
        array.append(inp_img)
    return   



def to_npy_array(folder_path, array, file_count):
    for i in range(0, file_count, 1):
        myfile = folder_path + str(i)+'.npy'
        img = np.load(myfile)
        img = normalize_image255(img)
        array.append(img)
        print('npyarray shape:', np.shape(array))
    return


# Load and pre-process the training data
input_images = []
output_images = []

# to_array('new1/1/fringeA/', input_images, 150)
# print('new1/1/fringeA')
# to_array('new1/1/wrap/', output_images, 150)
# print('new1/1/wrap')

to_png_array('/home/samir/Desktop/blender/pycode/scans/render', 'blenderimage0', input_images, IMAGECOUNT)
to_png_array('/home/samir/Desktop/blender/pycode/scans/render', 'im_wrap1' , output_images, IMAGECOUNT)


# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
input_images = np.expand_dims(input_images, -1)
output_images = np.expand_dims(output_images, -1)


print("input shape: {}".format(input_images.shape))
print("output shape: {}".format(output_images.shape))
print(len(input_images))

input_height = 160
input_width = 160

input_image = Input(shape=(input_height, input_width, 1))


# =============================================================================

x1 = Conv2D(50, (3, 3), padding='same')(input_image)
# x2 = BatchNormalization(axis=-1)(x1)
x3 = Activation('relu')(x1)

x4 = Conv2D(50, (3, 3), padding='same')(x3)
# x5 = BatchNormalization(axis=-1)(x4)
x6 = Activation('relu')(x4)
x7 = Conv2D(50, (3, 3), padding='same')(x6)
# x8 = BatchNormalization(axis=-1)(x7)
x9 = layers.add([x3, x7])
x10 = Activation('relu')(x9)

x11 = Conv2D(50, (3, 3), padding='same')(x10)
# x12 = BatchNormalization(axis=-1)(x11)
x13 = Activation('relu')(x11)
x14 = Conv2D(50, (3, 3), padding='same')(x13)
# x15 = BatchNormalization(axis=-1)(x14)
x16 = layers.add([x10, x14])
x17 = Activation('relu')(x16)

x18 = Conv2D(50, (3, 3), padding='same')(x17)
# x19 = BatchNormalization(axis=-1)(x18)
x20 = Activation('relu')(x18)
x21 = Conv2D(50, (3, 3), padding='same')(x20)
# x22 = BatchNormalization(axis=-1)(x21)
x23 = layers.add([x17, x21])
x24 = Activation('relu')(x23)

x25 = Conv2D(50, (3, 3), padding='same')(x24)
# x26 = BatchNormalization(axis=-1)(x25)
x27 = Activation('relu')(x25)
x28 = Conv2D(50, (3, 3), padding='same')(x27)
# x29 = BatchNormalization(axis=-1)(x28)
x30 = layers.add([x24, x28])
x31 = Activation('relu')(x30)

x32 = Conv2D(50, (3, 3), padding='same')(x31)
# x33 = BatchNormalization(axis=-1)(x32)
x34 = Activation('relu')(x32)
output_image = Conv2D(1, (3, 3), padding='same')(x34)


cnn1_model = Model(input_image, output_image)
cnn1_model.summary()


def compile_model(model):
    # model = Model(input_image, output_image)
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mse'])
    model.summary()
    return(model)



loss = []
val_loss = []
convweights = []

compile_model(cnn1_model)
model = cnn1_model


def load_model():
    model = keras.models.load_model(
        'models/cnnres01-300-modelwrap1+100-adam-noBN.h5')
    model.summary()
    return(model)


# model = load_model()

checkpointer = ModelCheckpoint(
    filepath="weights/weights.hdf5", verbose=1, save_best_only=True)


def fct_train():
    for epoch in range(EPOCHS):
        print('epoch #:', epoch)
        history_temp = model.fit(input_images, output_images,
                                 batch_size=4,
                                 epochs=1,
                                 validation_split=0.2,
                                 callbacks=[checkpointer])
        loss.append(history_temp.history['loss'][0])
        val_loss.append(history_temp.history['val_loss'][0])
        # convweights.append(model.layers[0].get_weights()[0].squeeze())


fct_train()


def plot():
    # Plot the training and validation losses
    plt.close('all')

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()
    plt.savefig('trainingvalidationlossgx.png')


plot()


def combImages(i1, i2, i3):
    new_img  = np.concatenate((i1, i2, i3), axis=1)
    return(new_img)


def DB_predict(i, x, y):
    predicted_img = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_img = predicted_img.squeeze()
    # cv2.imwrite('validate/'+str(i)+'filteredSync.png',
    #             (255.0*predicted_img).astype(np.uint8))
    # cv2.imwrite('validate/'+str(i)+'input.png',
    #             (255.0*x).astype(np.uint8))
    combo = combImages(255.0*x, 255.0*predicted_img, 255.0*y)
    # cv2.imwrite('validate/'+str(i)+'combo.png', (1.0*combo).astype(np.uint8))
    return(combo)


# get_my_file('inp/' + str(1)+'.png')
myfile = '/home/samir/Desktop/blender/pycode/scans/render' + str(1)+'/blenderimage0.png'
img = cv2.imread(myfile).astype(np.float32)
img = normalize_image255(img)
inp_img = make_grayscale(img)
combotot = combImages(inp_img, inp_img, inp_img)
for i in range(150, 220, 1):
    print(i)
    # get_my_file('inp/' + str(i)+'.png')
    myfile = '/home/samir/Desktop/blender/pycode/scans/render' + str(i)+'/blenderimage0.png'
    img = cv2.imread(myfile).astype(np.float32)
    img = normalize_image255(img)
    inp_img = make_grayscale(img)
    #get_my_file('out/' + str(i)+'.png')
    myfile = '/home/samir/Desktop/blender/pycode/scans/render' + str(i)+'/im_wrap1.png'
    img = cv2.imread(myfile).astype(np.float32)
    img = normalize_image255(img)
    out_img = make_grayscale(img)
    combo = DB_predict(i, inp_img, out_img)
    combotot = np.concatenate((combotot, combo), axis=0)
model.save('cnnpredict/models/cnnres01-220-modelwrap1'+'-200-adam-noBN.h5')
cv2.imwrite('validate/'+'cnnres01-220-wrap1'+'-200-adam-noBN.png',
            (1.0*combotot).astype(np.uint8))
