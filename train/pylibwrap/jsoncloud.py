
import argparse
import sys
import os
from PIL import Image
import cv2
import numpy as np
from pyntcloud import PyntCloud
import json

focalLength = 1290
centerX = 292
centerY = 286
scalingFactor = 5000  # 5000.0
rwidth = 400
rheight = 480


def generate_json_pointcloud(rgb_file, depth_file, json_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)
    rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
    depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    print(depth.mode)
    print(rgb.mode)
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "L":
        raise Exception("Depth image is not in intensity format")

    points = []
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u, v))
            Z = depth.getpixel((u, v)) *.22
            if Z == 0: continue
            Y = .22 * v
            X = .22 * u
            points.append(str(X) + ' ' + str(Y) + ' ' + str(Z))
            points.append(str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]))
    print('length is:', len(points))
    with open(json_file, 'w') as outfile:
        json.dump(points, outfile)
    outfile.close()


def generate_pointcloud(rgb_file, mask_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
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

