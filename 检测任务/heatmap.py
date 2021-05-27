import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import exp, log, sqrt, ceil


# 函数，用来画出高斯分布热力图，这里是以坐标为原点，5为半径来画的, 输入是x和y坐标点，输出是一个图像
def generate_heatmap(path, x, y):
    img = Image.open(path)
    img_x, img_y = img.size[1], img.size[0]
    image = np.zeros((img_x, img_y))
    # heatmap[32][24] = 1
    for i in range(len(x)):
        image[y[i]][x[i]] = 1
    heatmap = cv2.GaussianBlur(image, (7, 7), 0)
    am = np.amax(heatmap)
    # heatmap = heatmap * (255 / am)
    return heatmap


# 函数，用来返回热力图中热力值最大的坐标，输入是一个热力图图像,输出是图像中最大点的位置
def find_coordinate(heat_image):
    index = np.max(heat_image)
    coordinate = list(np.where(index == heat_image))
    y = coordinate[0]
    x = coordinate[1]
    return x, y