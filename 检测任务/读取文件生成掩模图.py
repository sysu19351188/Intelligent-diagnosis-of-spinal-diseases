import os
from txt读取 import *
from heatmap import *

'''编写函数，读取txt文件，并生成掩模图放置到指定文件夹中,
读入的参数为两个路径，第一个路径是txt文件所在的文件夹路径，第二个路径是最后储存掩膜的路径,
同时在得到掩膜的同时将原图像和掩膜图像变成统一大小，放入一个文件夹中，这样方便后续的网络训练'''


def generate_ym(or_path, ym_path):
    file_list = os.listdir(or_path)  # 读取txt文件所在的文件夹中所有的文件名，并放在fileList中
    # 这里是循环文件名列表，起点为1步长为2是因为我们的初始文件夹中第一个txt文件是在第二个位置，同时txt文件和jpg文件交叉出现
    for i in range(1, len(file_list), 2):
        txt_path = or_path + '/' + file_list[i]
        jpg_path = or_path + '/' + file_list[i - 1]
        data_list = txt_process(txt_path)  # txt文件处理，读取txt文件中的坐标点
        x = []
        y = []
        for j in range(len(data_list)):
            x.append(data_list[j][0])
            y.append(data_list[j][1])
        heat_map = generate_heatmap(jpg_path, x, y)
        heat_map = cv2.resize(heat_map, (256, 256))
        or_img = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)  # 这里将三通道的图转变为单通道的，我是打死也没想到原图是三通道的
        or_img = cv2.resize(or_img, (256, 256))
        cv2.imwrite(ym_path + '/' + file_list[i - 1] + '.jpg', or_img)
        cv2.imwrite(ym_path + '/' + file_list[i] + '.jpg', heat_map)


path1 = r"D:\pycharm1\脊柱疾病智能诊断\test\data\set"
save_path = r"D:\pycharm1"
generate_ym(path1, save_path)
