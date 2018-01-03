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

if __name__ == '__main__':
    im = array(Image.open("empire.jpg").convert("L"))
    harris_im = harris.compute_harris_response(im)
    filtered_coord_s = harris.get_harris_points(harris_im, 6, threshold=0.01)
    harris.plot_harris_points(im, filtered_coord_s)





