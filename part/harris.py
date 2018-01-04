# coding=utf-8
"""
!/usr/bin/env python
@Company : 元王有限元科技
@Author  : 陈彪
@Time    : 2018/1/3 16:27
@File    : harris.py
@Function: Harris角点检测器
"""

from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters

def compute_harris_response(im, sigma=3):
    """在一副灰度图像中，对每个像素计算Harris角点检测器响应函数"""

    # 计算导数
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    # 计算Harris矩阵的分量
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # 计算特征值和迹
    W_det = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return W_det / Wtr

def get_harris_points(harris_im, min_dist=10, threshold=0.1):
    """从一副Harris响应图像中返回角点。min_dist为分隔角点和图像边界的最小像素数目"""

    # 寻找高于阈值的候选角点
    corner_threshold = harris_im.max() * threshold
    harris_imt = (harris_im > corner_threshold ) * 1

    # 得到候选点的坐标
    coord_s = array(harris_imt.nonzero()).T

    # 以及他们的Harris响应值
    candidate_values = [harris_im[c[0], c[1]] for c in coord_s]

    # 对候选点按照Harris响应值进行排序
    index = argsort(candidate_values)

    # 将可行点的位置保持到数组中
    allowed_locations = zeros(harris_im.shape)
    allowed_locations[min_dist: -min_dist, min_dist: -min_dist] = 1

    # 按照min_distance原则，选择最佳Harris点
    filtered_coord_s = []
    for i in index:
        if allowed_locations[coord_s[i, 0], coord_s[i, 1]] == 1:
            filtered_coord_s.append(coord_s[i])
            allowed_locations[(coord_s[i, 0]-min_dist): (coord_s[i, 0]+min_dist),
            (coord_s[i, 1]-min_dist): (coord_s[i, 1]+min_dist)] = 0

    return filtered_coord_s

def plot_harris_points(image, filtered_coord_s):
    """绘制图像中检测到的角点"""

    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coord_s], [p[0] for p in filtered_coord_s], '*')
    axis('off')
    show()

def get_descriptors(image, filtered_coord_s, wid=5):
    """对于每个返回的点，返回点周围2*wid+1个像素的值（假设选取点的min_distance>wid）"""

    desc = []
    for coord in filtered_coord_s:
        patch = image[coord[0] - wid: coord[0] + wid + 1,
                coord[1] - wid: coord[1] + wid + 1].flatten()
        desc.append(patch)

    return desc

def match(desc1, desc2, threshold=0.5):
    """对于第一幅图像中的每个角点的描述子，使用归一化互相关，选取它在第二幅图像中的匹配角点"""

    n = len(desc1[0])

    # 点对的距离
    d = -ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = argsort(-d)
    match_scores = ndx[:, 0]

    return match_scores

def match_two_sided(desc1, desc2, threshold=0.5):
    """两边对称版本的match()"""

    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = where(matches_12 >= 0)[0]

    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12

def append_images(im1, im2):
    """返回将两幅图像并排拼接成的一副新图像"""

    # 选取具有最少行数的图像，然后填充足够的空行
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1-rows2, im2.shape[1]))), axis=0)

    return concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, loc1, loc2, match_scores, show_below=True):
    """显示一副带有连接匹配之间连线的图片
        输入：im1，im2（数组图像），loc1，loc2（特征位置），match_scores（match（）的输出）
        show_below（如果图像应该是显示匹配的下方）"""

    im3 = append_images(im1, im2)
    if show_below:
        im3 = vstack((im3, im3))

    imshow(im3)

    cols1 = im1.shape[1]
    for i, m in enumerate(match_scores):
        if m > 0:
            plot([loc1[i][1], loc2[m][1]+cols1], [loc1[i][0], loc2[m][0]], 'c')
        axis("off")

