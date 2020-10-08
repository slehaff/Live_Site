import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math
import os.path
from os import path
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
import argparse
import sys
# import myglobals





PI = np.pi



high_freq = 13
low_freq = 1
H = 160
W = 160


def resize(img, w,h):
    print(img.shape)
    img = img[0:160,0:160]
    print( img.shape )
    return(img)




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



def load_H_model():
    model = keras.models.load_model('/home/samir/dblive/cnnpredict/models/cnnres01-220-modelwrap1'+'-200-adam-noBN.h5')
    return(model)




def load_L_model():
    model = keras.models.load_model('/home/samir/dblive/cnnpredict/models/UNmodels/UNet02-224-wrap-kdata'+'-200-adam-noBN.h5')
    return(model)


def makemonohigh(folder):
    high = folder + 'high.png'
    colorhigh = cv2.imread(high, 1)
    monohigh = make_grayscale(colorhigh)
    cv2.imwrite(folder+'monohigh.png', monohigh)
    return


def mask(folder):
    color = folder + 'blendertexture.png'
    print('color:', color)
    img1 = np.zeros((H, W), dtype=np.float)
    img1 = cv2.imread(color, 1).astype(np.float32)
    gray = make_grayscale(img1)


    black = folder + 'blenderblack.png'
    img2 = np.zeros((H, W), dtype=np.float)
    img2 = cv2.imread(black, 0).astype(np.float32)
    diff1 = np.subtract(gray, .5*img2)
    mask =  np.zeros((H, W), dtype=np.float)
    for i in range(H):
        for j in range(W):
            if (diff1[i,j]<50):
                mask[i,j]= True
    np.save( folder+ 'mask.npy', mask, allow_pickle=False)
    cv2.imwrite( folder+ 'mask.png', 128*mask)
    return(mask)


def DB_predict( model, x):
    predicted_img = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_img = predicted_img.squeeze()
    # keras.backend.clear_session()
    return(predicted_img)




def nnHprocess(folder):
    high = folder + 'blenderimage0.png'
    image = cv2.imread(high, 1).astype(np.float32)
    inp_1 = normalize_image255(image)
    inp_1 = make_grayscale(inp_1)
    # Hmodel = load_H_model()

    start = time.time()
    predicted_img = DB_predict(Hmodel, inp_1)
    end = time.time()
    print('elapsed high:', end-start)

    mask = np.load(folder+'mask.npy')
    wrapH = np.multiply(np.logical_not(mask), predicted_img)
    wrapH = resize(wrapH, W, H)
    np.save(folder + 'cnnwrap1.npy', wrapH, allow_pickle=False)
    cv2.imwrite( folder + 'cnnwrap1.png',255*wrapH)
    return  #(predicted_img[0], predicted_img[1])

def nnLprocess(folder):
    high = folder + 'cnnwrap1.png'
    image = cv2.imread(high, 1).astype(np.float32)
    inp_1 = normalize_image255(image)
    inp_1 = make_grayscale(inp_1)
    # Hmodel = load_H_model()

    start = time.time()
    predicted_img = DB_predict(Lmodel, inp_1)
    end = time.time()
    print('elapsed low:', end-start)

    mask = np.load(folder+'mask.npy')
    kdata = np.multiply(np.logical_not(mask), predicted_img)
    kdatay = 255/6*kdata
    kdatay = np.round(kdatay)
    print('kdatay:', kdatay[::40, ::40])
    np.save(folder + 'nnkdata.npy', kdatay, allow_pickle=False)
    cv2.imwrite( folder + 'nnkdata.png',255*kdata)
    print('255*kdata:', 255*kdata[::40, ::40])


    return  #(predicted_img[0], predicted_img[1])
  

def unwrap_k(folder):
    kdata = np.zeros((H, W), dtype=np.float64)
    wraphigh = np.zeros((H, W), dtype=np.float64)
    unwrapdata = np.zeros((H, W), dtype=np.float64)
    kdata = np.load(folder + '/nnkdata.npy')

    # wraplow = resize(wraplow, W, H)  # To be continued
    wraphigh = 6.4*np.load(folder + '/cnnwrap1.npy')
    print('highrange=', np.ptp(wraphigh), np.max(wraphigh), np.min(wraphigh) )
    print('kdatarange=', np.ptp(kdata), np.max(kdata), np.min(kdata) )
    # wraphigh = normalize_image(wraphigh)
    # wraphigh = 2*PI*wraphigh
    # print('highrange=', np.ptp(wraphigh), np.max(wraphigh), np.min(wraphigh) )
    unwrapdata = np.add(wraphigh, np.multiply(2*PI,kdata) )
    wr_save = folder + '/nnkunwrap.npy'
    np.save(wr_save, unwrapdata, allow_pickle=False)
    cv2.imwrite(folder + '/nnkunwrap.png', 1.0*unwrapdata)




def makeDDbase(count):
    print('start')
    phibase = np.zeros((H, W,count), dtype=np.float64)
    for u in range(W):
        for v in range(H):
            print('u=', u, 'v=', v)
            for i in range(count):
                # print('i=',i)
                folder = '/home/samir/Desktop/blender/pycode/scanplanes/render'+ str(i)+'/'
                unwrap = np.zeros((H, W), dtype=np.float64)
                unwrap = np.load(folder+'unwrap.npy')   
                phibase[u,v,i] = unwrap[u,v]
    folder = '/home/samir/Desktop/blender/pycode/scanplanes/'
    wr_save = folder + 'DDbase.npy'
    np.save(wr_save, phibase, allow_pickle=False)


def makeDepth( folder, basecount):
    basefile = '/home/samir/Desktop/blender/pycode/scanplanes/DDbase.npy'
    DBase = np.load(basefile)
    unwrap = np.load(folder+'/nnkunwrap.npy' )
    # print('DBase:', np.amax(DBase), np.amin(DBase))
    # print('unwrap:', np.amax(unwrap), np.amin(unwrap))
    depth = np.zeros((H, W), dtype=np.float64)
    for i in range(W):
        # print('i:', i)
        for j in range(H):
            s=0
            for s in range(basecount-1):
                if (abs(unwrap[i,j]-DBase[i,j,s])<.15):
                    break
                else:
                    s+=1

            # print(i,j,unwrap[i,j],DBase[i,j,s])
            depth[i,j]=s
            # print('found:',i,j, unwrap[i,j], DBase[i,j,s],s)
    
    # print('depth:', np.amax(depth), np.amin(depth))
    # print(depth)
    im_depth = depth# np.max(unwrapdata)*255)
    cv2.imwrite(folder + 'nndepth.png', im_depth)



    


def generate_pointcloud(rgb_file, mask_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    print(rgb_file)
    rgb = Image.open(rgb_file)
    # depth = Image.open(depth_file)
    depth = Image.open(depth_file).convert('I')
    mask = Image.open(mask_file).convert('I')

    # if rgb.size != depth.size:
    #     raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")


    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color =   rgb.getpixel((u,v))
            # Z = depth.getpixel((u,v)) / scalingFactor
            # if Z==0: continue
            # X = (u - centerX) * Z / focalLength
            # Y = (v - centerY) * Z / focalLength
            if (mask.getpixel((u,v))<55):
                Z = depth.getpixel((u, v))*.22 
                if Z == 0: continue
                Y = .22 * v
                X = .22 * u
                points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()





def depth(scanfolder, count, basecount):
    for i in range(count):
        print('progress:', str(i))
        folder = '/home/samir/Desktop/blender/pycode/'+scanfolder+'/render'+ str(i)+'/'
        makeDepth(folder, basecount)


def nndepth(scanfolder, count, basecount):
    for i in range(count):
        print('progress:', str(i))
        folder = '/home/samir/Desktop/blender/pycode/'+scanfolder+'/'+ str(i)+'/'
        makeDepth(folder, basecount)


def unw(scanfolder, count):
    for i in range(count):
        print('start')

        folder = '/home/samir/Desktop/blender/pycode/'+scanfolder+'/render'+ str(i)+'/'
        print(folder)
        if path.exists(folder):
            unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )


def makeclouds(scanfolder, count):
     for i in range(count):
        print('start')
        folder = '/home/samir/Desktop/blender/pycode/'+scanfolder+'/render'+ str(i)+'/'
        print(folder)
        if path.exists(folder):
            # generate_pointcloud(folder + 'blendertexture.png', folder + '5mask.png' , folder + 'im_wrap1.png', folder +'pointcl-high.ply')
            # generate_pointcloud(folder + 'blendertexture.png', folder + '-1mask.png' , folder + 'im_wrap2.png', folder +'pointcl-low.ply')
            generate_pointcloud(folder + 'blendertexture.png', folder + '-1mask.png', folder + 'nnkunwrap.png', folder +'pointcl-unw.ply')
            generate_pointcloud(folder + 'blendertexture.png', folder + 'mask.png', folder + 'nndepth.png', folder +'pointcl-depth.ply')



####################################################################################################################

# folder = '/home/samir/serverless/new1-469/1/fringeA/' + str(i)+'.png'
folder = '/home/samir/Desktop/blender/pycode/inputscans/render'

Lmodel = load_L_model()
Hmodel = load_H_model()

for i in range(30):
    folder = '/home/samir/Desktop/blender/pycode/inputscans/render'
    print('i:', i)
    mask(folder+str(i)+'/')
    # unwrap_k(folder + str(i)+'/')
    # makemonohigh(folder+'i')
    nnHprocess(folder + str(i)+'/')
    nnLprocess(folder+str(i)+'/')
    unwrap_k(folder+str(i)+'/')
    makeDepth(folder+ str(i)+ '/', 199)
    folder=folder + str(i) +'/'
    generate_pointcloud(folder + 'blendertexture.png', folder + 'mask.png', folder + 'nndepth.png', folder +'pointcl-nndepth.ply')
    # generate_pointcloud(folder + 'blendertexture.png', folder + 'mask.png', folder + 'unwrap.png', folder +'pointcl-unw.ply')

    # unw('scans', 44)
    # depth('scans', 44, 199)
    # makeclouds('scans', 44)
    # print('done!')





#========================================================= Parking Lot============================================================


# def unwrap_r(low_f_file, high_f_file, folder):
#     filelow = folder + low_f_file
#     filehigh = folder +  high_f_file
#     wraplow = np.zeros((rheight, rwidth), dtype=np.float64)
#     wraphigh = np.zeros((rheight, rwidth), dtype=np.float64)
#     unwrapdata = np.zeros((rheight, rwidth), dtype=np.float64)
#     im_unwrap = np.zeros((rheight, rwidth), dtype=np.float64)
#     wraplow = np.load(filelow)  # To be continued
#     wraphigh = np.load(filehigh)
#     print('highrange=', np.ptp(wraphigh), np.max(wraphigh), np.min(wraphigh) )
#     print('lowrange=', np.ptp(wraplow), np.max(wraplow), np.min(wraplow) )
#     # print('high:', wraphigh)
#     # print('low:', wraplow)
    
#     unwrapdata = np.zeros((rheight, rwidth), dtype=np.float64)
#     kdata = np.zeros((rheight, rwidth), dtype=np.int64)
#     # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
#     # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
#     for i in range(rheight):
#         for j in range(rwidth):
#             kdata[i, j] = round((high_freq/low_freq * (wraplow[i, j])- wraphigh[i, j])/(2*PI))
#             # unwrapdata[i,j] = .1*(1.1*wraphigh[i, j]/np.max(wraphigh) +2*PI* kdata[i, j]/np.max(wraphigh))
#             # unwrapdata[i,j] = 1*(1*wraphigh[i, j] +2*PI* kdata[i, j])
#     unwrapdata = np.add(wraphigh, np.multiply(2*PI,kdata) )
#     print('kdata:', np.ptp(np.multiply(1,kdata)))
#     print('unwrap:', np.ptp(unwrapdata))
#     # print("I'm in unwrap_r")
#     print('kdata:', kdata[::40, ::40])
#     wr_save = folder + 'unwrap.npy'
#     np.save(wr_save, unwrapdata, allow_pickle=False)
#     maxval = np.amax(unwrapdata)
#     print('maxval:', maxval)
#     # im_unwrap = 255*unwrapdata/ maxval# np.max(unwrapdata)*255)
#     im_unwrap = 3*unwrapdata# np.max(unwrapdata)*255)
#     # unwrapdata/np.max(unwrapdata)*255
#     cv2.imwrite(folder + 'unwrap.png', im_unwrap)
    # cv2.imwrite(folder + 'kdata.png', np.multiply(2*PI,kdata))


# def unwrap(request):
#     # folder = ScanFolder.objects.last().folderName
#     folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/'
#     ref_folder = '/home/samir/db2/scan/static/scan_folder/scan_ref_folder'
#     three_folder = '/home/samir/db2/3D/static/3scan_folder'
#     unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
#     deduct_ref('unwrap.npy', 'unwrap.npy', folder, ref_folder)
#     # generate_color_pointcloud(folder + 'image1.png', folder + '/abs_unwrap.png', folder + '/pointcl.ply')
#     generate_json_pointcloud(folder + 'image1.png', folder +
#                              '/abs_unwrap.png', three_folder + '/pointcl.json')
#     return render(request, 'scantemplate.html')



# def unwrap_r(folder):
#     filelow = folder + '/scan_wrap2.npy'
#     filehigh = folder +  '/scan_wrap1.npy'
#     wraplow = np.zeros((H, W), dtype=np.float64)
#     wraphigh = np.zeros((H, W), dtype=np.float64)
#     unwrapdata = np.zeros((H, W), dtype=np.float64)
#     im_unwrap = np.zeros((H, W), dtype=np.float64)
#     wraplow = np.load(filelow)
#     wraplow = resize(wraplow, W, H)  # To be continued
#     wraphigh = np.load(filehigh)
#     wraphigh = resize(wraphigh, W, H)
#     print('highrange=', np.ptp(wraphigh), np.max(wraphigh), np.min(wraphigh) )
#     print('lowrange=', np.ptp(wraplow), np.max(wraplow), np.min(wraplow) )
#     # print('high:', wraphigh)
#     # print('low:', wraplow)
    
#     unwrapdata = np.zeros((H, W), dtype=np.float64)
#     kdata = np.zeros((H, W), dtype=np.int64)
#     # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
#     # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
#     for i in range(H):
#         for j in range(W):
#             kdata[i, j] = round((high_freq/low_freq * (wraplow[i, j])- wraphigh[i, j])/(2*PI))
#             # unwrapdata[i,j] = .1*(1.1*wraphigh[i, j]/np.max(wraphigh) +2*PI* kdata[i, j]/np.max(wraphigh))
#             # unwrapdata[i,j] = 1*(1*wraphigh[i, j] +2*PI* kdata[i, j])
#     unwrapdata = np.add(wraphigh, np.multiply(2*PI,kdata) )
#     print('kdata:', np.ptp(np.multiply(1,kdata)))
#     print('unwrap:', np.ptp(unwrapdata))
#     # print("I'm in unwrap_r")
#     print('kdata:', kdata[::40, ::40])
#     wr_save = folder + '/unwrap.npy'
#     np.save(wr_save, unwrapdata, allow_pickle=False)
#     maxval = np.amax(unwrapdata)
#     print('maxval:', maxval)
#     # im_unwrap = 255*unwrapdata/ maxval# np.max(unwrapdata)*255)
#     im_unwrap = 3*unwrapdata# np.max(unwrapdata)*255)
#     # unwrapdata/np.max(unwrapdata)*255
#     cv2.imwrite(folder + '/unwrap.png', im_unwrap)
#     cv2.imwrite(folder + '/kdata.png', np.multiply(2*PI,kdata))
#     np.save(folder+'/kdata.npy', kdata, allow_pickle=False )




# def makeDepth(folder, basecount):
#     basefile = '/home/samir/Desktop/blender/pycode/scanplanes/DDbase.npy'
#     DBase = np.load(basefile)
#     # unwrap = np.load(folder+'unwrap.npy' )
#     # print(unwrap.shape)
#     unwrap = cv2.imread(folder + 'nnkunwrap.png', 1).astype(np.float32)
#     unwrap = cv2.cvtColor(unwrap, cv2.COLOR_BGR2GRAY)
#     print(unwrap.shape)
#     # print('DBase:', np.amax(DBase), np.amin(DBase))
#     # print('unwrap:', np.amax(unwrap), np.amin(unwrap))
#     depth = np.zeros((H, W), dtype=np.float64)
#     for i in range(W):
#         # print('i:', i)
#         for j in range(H):
#             s=0
#             for s in range(basecount-1):
#                 if (abs(unwrap[i,j]-DBase[i,j,s])<.15): #used -10 to offset scans449 to avoiod saturation
#                     break
#                 else:
#                     s+=1

#             # print(i,j,unwrap[i,j],DBase[i,j,s])
#             depth[i,j]=s
#             # print('found:',i,j, unwrap[i,j], DBase[i,j,s],s)
    
#     # print('depth:', np.amax(depth), np.amin(depth))
#     im_depth = depth# np.max(unwrapdata)*255)
#     cv2.imwrite(folder + 'nndepth.png', im_depth)
    
