# coding=utf-8
"""
!/usr/bin/env python
@Company : 元王有限元科技
@Author  : 陈彪
@Time    : 2018/1/3 8:16
@File    : demo.py
@Function: 
"""
from numpy import random
from pylab import *
from scipy.ndimage import filters
from PIL import Image
from part import harris
from base import rof
from part import sift

if __name__ == '__main__':
    im_name = "empire.jpg"
    im = array(Image.open(im_name).convert("L"))
    sift.process_image(im_name, 'empire.sift')
    l1, d1 = sift.read_features_from_file("empire.sift")

    figure()
    gray()
    sift.plot_features(im, l1)
    print("done")
    show()





