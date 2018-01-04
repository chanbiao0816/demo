# coding=utf-8
"""
!/usr/bin/env python
@Company : 元王有限元科技
@Author  : 陈彪
@Time    : 2018/1/4 8:01
@File    : sift.py
@Function: 尺度不变特征变换
"""
import os
from PIL import Image
from numpy import *
from pylab import *

def process_image(image_name, result_name, params="--edge-thresh 10 --peak-thresh 5"):
    """处理一幅图像，然后将结果保存在文件中"""

    if image_name[-3:] != 'pgm':
        # 创建一个pgm文件
        im = Image.open(image_name).convert("L")
        im.save("tmp.pgm")
        image_name = "tmp.pgm"

    sift_path = "D:/vlfeat-0.9.20/bin/win64/sift "
    cmd = str(sift_path + image_name + " --output=" + result_name + " " + params)
    os.system(cmd)
    print("processed", image_name, "to", result_name)

def read_features_from_file(file_name):
    """读取特征属性值，然后将其以矩阵的形式返回"""

    f = loadtxt(file_name)
    return f[:, :4], f[:, 4:]   # 特征位置，描述子

def write_features_to_file(file_name, loc, desc):
    """将特征位置和描述子保存到文件中"""
    savetxt(file_name, hstack((loc, desc)))

def plot_features(im, loc, circle=False):
    """显示带有特征的图像
        输入：im（数组图像），loc（每个特征的行、列、尺度和朝向）"""

    def draw_circle(c, r):
        t = arange(0, 1.01, .01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)

        imshow(im)

    if circle:
        for p in loc:
            draw_circle(p[:2], p[2])
    else:
        plot(loc[:, 0], loc[:, 1], 'ob')
    axis("off")


def match(desc1, desc2):
    """对于第一幅图像中的每个描述子，选取其在第二幅图像中的匹配"""
    desc1 = array([d/linalg.norm(d) for d in desc1])
    desc2 = array([d/linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    match_scores = zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T    # 预先计算矩阵转置
    for i in range(desc1_size[0]):
        dot_prods = dot(desc1[i, :], desc2t)     # 向量点乘
        dpt_prods = 0.9999 * dot_prods
        # 返余弦和反排序，返回第二幅图像中特征的索引
        index = argsort(arccos(dot_prods))

        # 检查最近邻的角度是否小于dist_ratio乘以第二近邻的角度
        if arccos(dot_prods)[index[0]] < dist_ratio * arccos(dot_prods)[index[1]]:
            match_scores[i] = int(index[0])

        return match_scores


