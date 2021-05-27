import os
from txt读取 import *
from heatmap import *

'''编写函数，读取文件夹中的内容，同时按照坐标点对图片进行切割，分割出11块对应的部分，这样方便分类任务的进行'''


def generate_qg(or_path, ym_path):
    file_list = os.listdir(or_path)  # 读取txt文件所在的文件夹中所有的文件名，并放在fileList中
    # 这里是循环文件名列表，起点为1步长为2是因为我们的初始文件夹中第一个txt文件是在第二个位置，同时txt文件和jpg文件交叉出现
    for i in range(1, len(file_list), 2):
        txt_path = or_path + '/' + file_list[i]
        jpg_path = or_path + '/' + file_list[i - 1]
        data_list = txt_process(txt_path)  # txt文件处理，读取txt文件中的坐标点
        print(jpg_path)
        img = Image.open(jpg_path)
        img_x, img_y = img.size[1], img.size[0]
        img1 = cv2.imread(jpg_path)
        img = cv2.resize(img1, (256, 256))
        name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        for j in range(len(data_list)):
            x_ = int(data_list[j][0] / img_x * 256)
            y_ = int(data_list[j][1] / img_x * 256)
            or_img = img[y_-10: y_+10, x_-25: x_+25]
            print(data_list)
            print("-----------------------")
            if j%2==0:
                if data_list[j][2]=='v1':
                   cv2.imwrite(ym_path + '/inter_v1/' + file_list[i - 1] + name[j] + '.jpg', or_img)
                if data_list[j][2]=='v2':
                   cv2.imwrite(ym_path + '/inter_v2/' + file_list[i - 1] + name[j] + '.jpg', or_img)
                if data_list[j][2]=='v3':
                   cv2.imwrite(ym_path + '/inter_v3/' + file_list[i - 1] + name[j] + '.jpg', or_img)
                if data_list[j][2]=='v4':
                   cv2.imwrite(ym_path + '/inter_v4/' + file_list[i - 1] + name[j] + '.jpg', or_img)
                if data_list[j][2]=='v5':
                   cv2.imwrite(ym_path + '/inter_v5/' + file_list[i - 1] + name[j] + '.jpg', or_img)
            if j%2==1:
                if data_list[j][2] == 'v1':
                    cv2.imwrite(ym_path + '/cone_v1/' + file_list[i - 1] + name[j] + '.jpg', or_img)
                if data_list[j][2] == 'v2':
                    cv2.imwrite(ym_path + '/cone_v2/' + file_list[i - 1] + name[j] + '.jpg', or_img)
            with open(r"C:\Users\biu boon\Desktop\sdfgh.txt", "a", encoding='utf-8') as f:
                f.write(str(data_list[j][2]))
                f.close()
        with open(r"C:\Users\biu boon\Desktop\sdfgh.txt", "a", encoding='utf-8') as f:
            f.write('\n')
            f.close()
path1 = r"C:\Users\biu boon\Desktop\data"
path2 = r"D:\pycharm1\set"
generate_qg(path1, path2)

