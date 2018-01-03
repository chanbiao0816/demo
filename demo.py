# coding=utf-8
"""
!/usr/bin/env python
@Company : 元王有限元科技
@Author  : 陈彪
@Time    : 2018/1/3 8:16
@File    : demo.py
@Function: 
"""
from PIL import Image
from numpy import *
from pylab import *
from numpy import random
from scipy.ndimage import filters
import rof

im = zeros((500, 500))
im[100:400, 100:400] = 128
im[200:300, 200:300] = 255

im = im + 30 * random.standard_normal((500, 500))

U, T = rof.de_noise(im, im)
G = filters.gaussian_filter(im, 10)

# 保存生成结果
from scipy.misc import imsave
imsave('synth_rof.pdf', U)
imsave('synth_gaussian.pdf', G)





