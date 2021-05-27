# 导入所需工具包
import matplotlib
from keras.models import load_model
import argparse
import pickle
import cv2
from keras.layers import Dropout
from keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD
from keras import initializers
from keras import regularizers
import fpaths # 处理图像路径
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
import random
import pickle
import cv2
import os

#开始读取数据
data = []
labels = []
# 拿到图像数据路径，方便后续读取
imagePaths = sorted(list(fpaths.list_images('./dataset')))
random.seed(42)
random.shuffle(imagePaths)
# 遍历读取数据
for imagePath in imagePaths:
    # 读取图像数据，由于使用神经网络，需要给拉平成一维
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)
    # 读取标签
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# 对图像数据做scale操作
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# 切分数据集
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)
# 转换标签为one-hot encoding格式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
print("------")
# 构造网络模型结构：本次为3072-128-64-3
model = Sequential()
# kernel_regularizer=regularizers.l2(0.01) L2正则化项
# initializers.TruncatedNormal 初始化参数方法，截断高斯分布
model.add(Dense(128, input_shape=(3072,), activation="relu" ,kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu",kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(lb.classes_), activation="softmax",kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),kernel_regularizer=regularizers.l2(0.01)))
# 初始化参数
INIT_LR = 0.001
EPOCHS = 2000
# 模型编译
#准备训练网络
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
# 拟合模型
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
# 测试网络模型
#开始评估模型
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))
# 保存模型到本地
model.save('././output/simple_nn.model')
f = open('./output/simple_nn_lb.pickle', "wb") # 保存标签数据
f.write(pickle.dumps(lb))
f.close()
# 导入所需工具包

#加载模型开始预测
# 加载测试数据并进行相同预处理操作
image = cv2.imread('./cs_image/panda.jpg')
output = image.copy()
image = cv2.resize(image, (32, 32))
# scale图像数据
image = image.astype("float") / 255.0
# 对图像进行拉平操作
image = image.flatten()
image = image.reshape((1, image.shape[0]))
# 读取模型和标签
model = load_model('./output/simple_n.model')
lb = pickle.loads(open('./output/simple_lb.pickle', "rb").read())
# 预测
preds = model.predict(image)
# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
# 在图像中把结果画出来
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
# 绘图
cv2.imshow("Image", output)
cv2.waitKey(0)