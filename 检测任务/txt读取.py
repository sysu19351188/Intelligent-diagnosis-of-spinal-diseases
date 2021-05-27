def txt_process(filename):  # 这个函数用来处理txt里的数据，将各坐标点和标签提取出来，然后将这些数据放入一个list中
    count = 1
    all_count = 1
    data_list = []
    with open(filename, "r") as ins:  # 打开文件
        for line in ins:
            all_count += 1
            line = line.replace(" ", "")  # 将空格和换行去掉
            line = line.replace("\n", "")
            if 'vertebra' in line and '-' not in line:
                index_label = line.index('vertebra')
                index_y = line.index(',{')
                index_x = line.index(',')
                num_x = int(line[0:index_x])
                num_y = int(line[index_x + 1:index_y])
                label = line[index_label + 11:index_label + 13]
            elif 'disc' in line and '-' in line:
                index_label = line.index('disc')
                index_y = line.index(',{')
                index_x = line.index(',')
                num_x = int(line[0:index_x])
                num_y = int(line[index_x + 1:index_y])
                label = line[index_label + 7:index_label + 9]
            else:
                continue
            data_list.append((num_x, num_y, label))
            data_list = sorted(data_list, key=lambda x: x[1])
    return data_list
a = txt_process(r"D:\a_temp\脊柱疾病智能诊断\test\data\study45.txt")
print(a)