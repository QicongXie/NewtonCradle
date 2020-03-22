"""
---------------------------------------------------------------------
-- Author: Xie Qicong (1744792309@qq.com)
---------------------------------------------------------------------

Newton's Cradle

"""

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class NewtonCradle(object):
    def __init__(self, N=5, R=0.0111, m=0.04055, L=0.125, E=2e11, o=0.33, kr=3470, T=50, dT=0.00001, g=9.79, kn=6.8e-4,
                 init_angles=None):
        """

        :param N: 小球的个数
        :param R: 小球的半径
        :param m: 小球的质量
        :param L: 摆绳的长度
        :param E: 杨氏模量
        :param o: 泊松比
        :param kr: 粘弹性损失系数
        :param T: 仿真时间
        :param dT: 时间步长
        :param g: 重力加速度
        :param kn: 空气阻力系数
        """
        self.N = N
        self.R = R
        self.m = m
        self.L = L
        self.E = E
        self.o = o
        self.kr = kr
        self.T = T
        self.dT = dT
        self.g = g
        self.kn = kn
        if init_angles is None:
            init_angles = [-60, 0, 0, 0, 0]
        self.scale = 1000
        self.fps = 120

        self.NT = round(T / dT)
        self.k = math.sqrt(2 * R) * 2e11 / (3 * (1 - math.pow(0.33, 2)))  # 碰撞劲度系数

        self.der_x = np.zeros((self.NT, self.N))  # 初始速度 0 m/s
        self.der_x[0, 0] = 0  # x,y方向上的速度都是0
        self.x = np.zeros((self.NT, self.N))
        self.x[0, :] = np.arange(0, 2 * self.R * self.N, 2 * self.R)
        for id in range(self.N):
            self.x[0, id] = self.x[0, id] + self.L * math.sin(math.pi / 180 * init_angles[id])
        self.f1 = np.zeros((self.NT, self.N))
        self.f2 = np.zeros((self.NT, self.N))  # ;%与前后球的重叠长度

        self.isCalc = False  # 表示还没计算过呢

    # def __str__(self):
    #
    #     return

    def sin(self, point, no, t=None):
        x = point
        result = (x - no * 2 * self.R) / self.L
        return result

    def cos(self, point1, no, t=None):
        result = math.sqrt(1 - self.sin(point1, no, t) ** 2)
        return result

    def TT(self, time, N):
        t = self.m * (self.der_x[time - 1, N] / self.cos(self.x[time, N], N, time)) ** 2 / self.L + math.fabs(
            self.m * self.g * self.cos(self.x[time, N], N, time))
        return t

    def calc(self):
        for t in tqdm(range(1, self.NT)):
            for i in range(0, self.N):  # 这边可以化简为矩阵计算
                self.x[t, i] = self.x[t - 1, i] + self.der_x[t - 1, i] * self.dT

            self.f1[t, self.N - 1] = ((2 * self.R - self.x[t, self.N - 1] + self.x[t, self.N - 2]) > 0) * (
                    2 * self.R - self.x[t, self.N - 1] + self.x[t, self.N - 2])
            self.f2[t, 0] = ((2 * self.R - self.x[t, 1] + self.x[t, 0]) > 0) * (
                    2 * self.R - self.x[t, 1] + self.x[t, 0])
            self.der_x[t, 0] = (self.k * (-math.pow(self.f2[t, 0], 1.5))
                                - self.TT(t, 0) * self.sin(self.x[t, 0], 0, t)
                                - self.kn * self.der_x[t - 1, 0]
                                - self.kr * (math.pow(self.f2[t, 0], 1.5) - math.pow(self.f2[t - 1, 0],
                                                                                     1.5)) / self.dT) / self.m * self.dT \
                               + self.der_x[t - 1, 0]

            self.der_x[t, self.N - 1] = (self.k * (math.pow(self.f1[t, self.N - 1], 1.5))
                                         - self.TT(t, self.N - 1) * self.sin(self.x[t, self.N - 1], self.N - 1, t)
                                         - self.kn * self.der_x[t - 1, self.N - 1]
                                         - self.kr * (math.pow(self.f1[t, self.N - 1], 1.5) - math.pow(
                        self.f1[t - 1, self.N - 1], 1.5)) / self.dT) / self.m * self.dT \
                                        + self.der_x[t - 1, self.N - 1]

            for i in range(1, self.N - 1):
                self.f1[t, i] = ((2 * self.R - self.x[t, i] + self.x[t, i - 1]) > 0) * (
                        2 * self.R - self.x[t, i] + self.x[t, i - 1])
                self.f2[t, i] = ((2 * self.R - self.x[t, i + 1] + self.x[t, i]) > 0) * (
                        2 * self.R - self.x[t, i + 1] + self.x[t, i])

                self.der_x[t, i] = (self.k * (math.pow(self.f1[t, i], 1.5) - math.pow(self.f2[t, i], 1.5))
                                    - self.TT(t, i) * self.sin(self.x[t, i], i, t)
                                    - self.kn * self.der_x[t - 1, i]
                                    - self.kr * (math.pow(self.f2[t, i], 1.5) - math.pow(self.f1[t, i], 1.5) - math.pow(
                            self.f2[t - 1, i], 1.5) + math.pow(self.f1[t - 1, i], 1.5)) / self.dT) / self.m * self.dT \
                                   + self.der_x[t - 1, i]

            self.isCalc = True

    def findEnvelope(self, X, window_size=100000, k=1):
        """

        :param window_size: 窗口长度
        :param k: 样条插值
        :return:
        """
        if not self.isCalc:
            raise
        a = X.reshape(-1, window_size)
        x_max = np.max(a, axis=1)
        yy = np.arange(0, self.NT / window_size) * self.dT * window_size + 0.5
        y = np.arange(0, self.NT) * self.dT

        ipo = spi.splrep(yy, x_max, k=k)  # 样本点导入，生成参数
        envelope = spi.splev(y, ipo)  # 根据观测点和样条参数，生成插值
        return envelope, y

    def calconeball(self, x_pos, L, bias):
        x_pos = x_pos - bias
        y = math.pow(x_pos, 2) / L
        x = x_pos * math.pow(1 - math.pow(x_pos / (2 * L), 2), 0.5)
        return (x, y)

    def calcpos(self, xpos, R, L):
        pos = []
        nums = len(xpos)
        for i in range(nums):
            pos.append(self.calconeball(xpos[i], L, i * 2 * R))
        return pos

    def plot(self):
        if not self.isCalc:
            raise
        y = np.arange(0, self.NT) * self.dT
        style = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
        for i in range(self.N):
            plt.plot(y, self.x[:, i], ls=style[i], lw=2, label="No . " + str(i))
        plt.legend()
        plt.show()

    def saveVideo(self, name="videos.mp4"):
        if not self.isCalc:
            raise
        size = (380, 320)
        print(name)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(name, fourcc, self.fps, size)

        speed = 1 / self.fps / self.dT

        for t in tqdm(range(int(self.x.shape[0] // speed))):
            calp = self.calcpos(self.x[t * int(speed)], self.R, self.L)
            img = np.zeros((320, 380, 3), np.uint8)  # 生成一个空灰度图像
            img.fill(255)
            xs = 120
            ys = 60
            for i in range(self.N):
                ptStart = (xs + int(self.scale * i * 2 * self.R), ys)
                ptEnd = (xs + int(self.scale * i * 2 * self.R) + int(self.scale * calp[i][0]),
                         ys + int(self.scale * (self.L - calp[i][1])))
                point_color = (0, 255, 0)  # BGR
                thickness = 1
                cv2.line(img, ptStart, ptEnd, point_color, thickness, cv2.LINE_AA)
                cv2.circle(img, ptEnd, int(self.scale * self.R), point_color, thickness, cv2.LINE_AA)
            videowriter.write(img)
        videowriter.release()
        print("finish\n")


def main():
    style = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    # 实验一：五个球绳子长度分别是 7.5     10 .    12.5    16      18 释放高度不变 假设是5cm
    # 换算出来的初始角度：               70.5   60       53      46.5     43.7
    print("===== 实验一 =====")
    lengths = [7.5, 10, 12.5, 16, 18]
    for i in range(5):
        nc = NewtonCradle(L=lengths[i] / 100)
        nc.calc()
        nc.saveVideo(name="length_" + str(lengths[i]) + ".mp4")
        e, xs = nc.findEnvelope(nc.x[:, 4] - 2 * 4 * nc.R)
        plt.plot(xs, e, ls=style[i], lw=1, label="绳子长度：" + str(lengths[i]))
        # np.save("length_" + str(lengths[i]), nc.x)
    plt.legend()
    plt.savefig("实验一.png")
    plt.clf()

    # 实验二 ： 绳子长度都是15cm 释放的角度变化 5  10 30 50 70
    print("===== 实验二 =====")
    angles = [5, 10, 30, 50, 70]
    for i in range(5):
        nc = NewtonCradle(init_angles=[-angles[i], 0, 0, 0, 0])
        nc.calc()
        nc.saveVideo(name="angle_" + str(angles[i]) + ".mp4")
        e, xs = nc.findEnvelope(nc.x[:, 4] - 2 * 4 * nc.R)
        plt.plot(xs, e, ls=style[i], lw=1, label="绳子角度：" + str(angles[i]))
        # np.save("angle_" + str(angles[i]),nc.x)
    plt.legend()
    plt.savefig("实验二.png")
    plt.clf()
    # 实验三：改变球的材料：
    #                                  E                      o                  kr
    # 钢球                  |    2e11         |    0.33        |      2000
    # 硬质玻璃球      |    1.9e11      |   0.25        |       2500
    # 有机玻璃球      |   0.55e11    |     0.23       |      2800
    #  塑料球             |   0.5e9         |     0.65       |      3500
    print("===== 实验三 =====")
    Es = [2e11, 1.9e11, 0.55e11, 0.5e9]
    Os = [0.33, 0.25, 0.23, 0.65]
    Krs = [2000, 2500, 2800, 3500]
    names = ["钢球", "硬质玻璃球", "有机玻璃球", "塑料球"]
    for i in range(4):
        nc = NewtonCradle(E=Es[i], o=Os[i], kr=Krs[i])
        nc.calc()
        nc.saveVideo(name="材料_" + str(names[i]) + ".mp4")
        e, xs = nc.findEnvelope(nc.x[:, 4] - 2 * 4 * nc.R)
        plt.plot(xs, e, ls=style[i], lw=1, label="小球材料：" + str(names[i]))
        # np.save("材料_" + str(names[i]) , nc.x)
    plt.legend()
    plt.savefig("实验三.png")
    plt.clf()

    # 实验四：改变小球的个数
    # 一个球碰五个球，一碰4，一碰3，一碰2，一碰1
    print("===== 实验四 =====")
    nums = [2, 3, 4, 5, 6]
    init_angle = [[-60, 0], [-60, 0, 0], [-60, 0, 0, 0], [-60, 0, 0, 0, 0], [-60, 0, 0, 0, 0, 0]]
    for i in range(5):
        nc = NewtonCradle(N=nums[i], init_angles=init_angle[i])
        nc.calc()
        nc.saveVideo(name="num_" + str(nums[i]) + ".mp4")
        e, xs = nc.findEnvelope(nc.x[:, nums[i] - 1] - 2 * (nums[i] - 1) * nc.R)
        plt.plot(xs, e, ls=style[i], lw=1, label="小球个数：" + str(nums[i]))
        # np.save("num_" + str(nums[i]),nc.x)
    plt.legend()
    plt.savefig("实验四.png")
    plt.clf()


if __name__ == '__main__':
    main()
