# coding=utf-8
"""
!/usr/bin/env python
@Company : 元王有限元科技
@Author  : 陈彪
@Time    : 2018/1/3 8:56
@File    : rof.py
@Function: 图像去噪（Rudin-Osher-Fatemi模型）
"""

from numpy import *

def de_noise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """使用A.Chambolle（2005）实现ROF模型
        输入：含有噪声的输入（灰度）图像、U的初始值、TV正则项权值、步长、停业条件
        输出：去燥和去除纹理后的图像、纹理残留"""
    m, n = im.shape

    # 初始化
    U = U_init
    Px = im # 对偶域的x分量
    Py = im # 对偶域的y分量
    error = 1

    while error > tolerance:
        U_old = U

        # 原始变量的梯度
        GradUx = roll(U, -1, axis=1) - U    # 变量U梯度的x分量
        GradUy = roll(U, -1, axis=0) - U    # 变量U梯度的y分量

        # 更新对偶变量
        PxNew = Px + (tau/tv_weight) * GradUx
        PyNew = Py + (tau/tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew**2 + PyNew**2))

        Px = PxNew/NormNew  # 更新x分量（对偶）
        Py = PyNew/NormNew  # 更新y分量（对偶）

        # 更新原始变量
        RxPx = roll(Px, 1, axis=1)  # 对x分量进行向右x轴平移
        RyPy = roll(Py, 1, axis=0)  # 对y分量进行向上y轴平移

        DivP = (Px - RxPx) + (Py - RyPy)    # 对偶域的散度
        U = im + tv_weight * DivP   # 更新原始变量

        # 更新误差
        error = linalg.norm(U - U_old)/sqrt(n*m)

    return U, im-U  # 去燥后的图像和纹理残余
