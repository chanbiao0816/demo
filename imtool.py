# coding=utf-8
"""
!/usr/bin/env python
@Company : 元王有限元科技
@Author  : 陈彪
@Time    : 2018/1/2 18:04
@File    : imtool.py
@Function: 
"""

from pylab import *
from PIL import Image
from numpy import *


def im_resize(im, sz):
    """使用 PIL 对象重新定义图像数组的大小"""
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def hist_eq(im, nbr_bins=256):
    """对一幅灰度图像进行直方图均衡化"""
    # 计算图像的直方图
    im_hist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = im_hist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # 归一化
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(im_list):
    """计算图像列表的平均图像"""

    # 打开第一幅图像，将其存储在浮点型数组中
    average_im = array(Image.open(im_list[0]), 'f')

    for im_name in im_list[1:]:
        try:
            average_im += array(Image.open(im_name))
        except:
            print(im_name + '...skiped')

    average_im /= len(im_list)
    # 返回uint8类型的平均图像
    return array(average_im, 'uint8')

def pca(X):
    """主成分分析：
        输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
        返回：投影矩阵（按照维度的重要性排序）、方差和均值"""

    # 获取维数
    num_data, dim = X.shape

    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA- 使用紧致技巧
        M = dot(X, X.T)     # 协方差矩阵
        e, EV = linalg.eigh(M)  # 特征值和特征向量
        tmp = dot(X.T, EV).T    # 这就是紧致技巧
        V = tmp[::-1]   # 由于最后的特征向量是我们所需要的，所以需要将其逆转


