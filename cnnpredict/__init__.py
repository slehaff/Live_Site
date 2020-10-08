# this module containns an input thread for reading input buffers from scanner upload
# The module reads n buffers over once 100 buffers have been received

import time
import myglobals
import threading
import cv2
import numpy as np
import os


def loopforever():
    folder = '/home/samir/Desktop/blender/pycode/inputscans/render'
    a=0
    b=0
    while(a==0):
        print(len(myglobals.inque), b)
        b+=1
        time.sleep(20)
        if (len(myglobals.inque)>100):
            for i in range(33):
                f = folder + str(i)
                os.mkdir(f)
                black = myglobals.inque.popleft()
                color = myglobals.inque.popleft()
                high = myglobals.inque.popleft()
                print('black', len(black))
                print('color', len(color))
                print('high', len(high))
                cv2.imwrite(f+'/blenderimage0.png', high)
                cv2.imwrite(f+'/blendertexture.png', color)
                cv2.imwrite(f+'/blenderblack.png', black)
                print(f)
t = threading.Thread(target=loopforever)

t.start()


