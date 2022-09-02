## openCV

### 介绍

**简介**：`OpenCV`（ 开源计算机视觉库：http://opencv.org ）是一个包含数百种计算机视觉算法的开源库

**官网**：https://opencv.org/

**官方文档**：https://docs.opencv.org/4.6.0/

**下载**：https://opencv.org/releases/

**教程**： https://www.bilibili.com/video/BV1Lq4y1Z7dm

### 安装

#### 使用pip+官方文件安装

**环境**：python3.9, pycharm

**安装相关库**：

```sh
# cmd输入
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install PIL-Tools -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**下载openCV 4.6.0官方库**：https://opencv.org/releases/

### 一个头像识别demo

**简介**：本项目使用opencv库选择器`haarcascade_frontalface_alt2.xml`，简单介绍了opencv人脸识别，模型训练的的主要过程

#### 调用图片

```python
import cv2 as cv

img = cv.imread('./img/1.png')
cv.imshow('read_img', img)
cv.waitKey(0)
cv.destroyAllWindows()
```

#### 调整图片灰度

```python
# 导入cv模块
import cv2 as cv
# 读取图片
img = cv.imread('./img/1.png')
# 灰度转换
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 显示灰度
cv.imshow('gray', gray_img)
# 保存灰度图片
cv.imwrite('./img/11.png', gray_img)
# 显示图片
cv.imshow('read_img', img)
# 等待
cv.waitKey(0)
# 释放内存
cv.destroyAllWindows()
```

#### 尺寸修改

```python
# 导入cv模块
import cv2 as cv
# 读取图片
img = cv.imread('./img/1.png')
# 修改尺寸
resize_img = cv.resize(img, dsize=(200, 200))
# 显示原图
cv.imshow('img', img)
# 显示修改的图
cv.imshow('resize_img', resize_img)
# 打印原图尺寸大小
print('未修改：', img.shape)
# 打印修改后的大小
print('修改后：', resize_img.shape)
# 等待
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
```

#### 绘制矩形

```python
# 导入cv模块
import cv2 as cv
# 读取图片
img = cv.imread('./img/1.png')
# 坐标
x, y, w, h = 100, 100, 100, 100
# 绘制矩形
cv.rectangle(img, (x, y, x+w, y+h), color=(0, 0, 255), thickness=1)
# 绘制圆形
cv.circle(img, center=(x+w,y+h), radius=100, color=(255,0,0),thickness=2)
# 显示
cv.imshow('re_img', img)
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
```

#### 人脸检测

```python
# 导入cv模块
import cv2 as cv
# 绘制函数
def face_detect_demo():
    gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary, 1.01, 5)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result', img)

# 读取图像
img =cv.imread('./img/1.png')
# 检测函数
face_detect_demo()
# 等待
while True:
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()

```

#### 摄像头(视频检测)

```python
# 导入cv模块
import cv2 as cv
# 绘制函数
def face_detect_demo(img):
    gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary, 1.01, 5)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result', img)
# 读取摄像头
cap = cv.VideoCapture(0)
# 读取视频
# cap = cv.VideoCapture('1.mp4')


# 等待
while True:
    flag, frame = cap.read()
    # 如果没有图像了，退出循环
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
# 释放摄像头
cap.release()
```

#### 人脸录入

```python
# 导入cv模块
import cv2 as cv
# 调用摄像头
cap = cv.VideoCapture(0)

num = 1

while(cap.isOpened()):  # 检测是否开启摄像头
    ret_flag, Vshow = cap.read()    # 得到每帧图像
    cv.imshow("Capture_test", Vshow)    # 显示图像
    k = cv.waitKey(1) & 0xFF    # 按键判断
    if k == ord('s'): # 保存
        cv.imwrite("./saveimg/"+str(num)+".name"+".jpg", Vshow)
        print("success to save"+str(num)+".jpg")
        print("-------------------")
        num += 1
    elif k == ord(' '): # 退出
        break

# 释放摄像头
cap.release()
# 释放内存
cv.destroyAllWindows()
```

#### 数据训练

```python
import os
import cv2
import sys
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    # 储存人脸数据
    facesSamples=[]
    # 储存姓名数据
    ids=[]
    # 储存图片信息
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片，灰度化。PIL有九种不同模式：1,L,P,RGB,RGBA,CMYK,YCbCr,I,F.
        PIL_img=Image.open(imagePath).convert('L')
        # 将图片转换为数组，以黑白生成
        img_numpy=np.array(PIL_img,'uint8')
        # 获取图片人脸特征,用分类器分类
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名
        id=int(os.path.split(imagePath)[1].split('.')[0])
        # 预防无面容照片
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
    # 打印脸部特征和id
    print('id', id)
    print('fs:', facesSamples)
    return facesSamples,ids

if __name__ == '__main__':
    # 图片路径
    path='./saveimg/'
    # 获取图像数组和id，标签数组和姓名
    faces,ids=getImageAndLabels(path)
    # 加载识别器
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces,np.array(ids))
    # 保存文件
    recognizer.write('trainer/trainer.yml')
```

#### 人脸识别

```python
import os
import cv2


# 加载训练数据集文件
recongizer = cv2.face.LBPHFaceRecognizer_create()
recongizer.read('trainer/trainer.yml')
names=[]
warningtime = 0
# 准备识别的图片
def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转换为灰度
    face_detector=cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml') # 加载模型
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        # 人脸识别
        ids, confidence = recongizer.predict(gray[y:y + h, x:x + w])
        # ids:标签, confidence:置信评分
        if confidence > 70:
            cv2.putText(img, 'unknow', (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,255),2)
        else:
            cv2.putText(img,str(names[ids-1]),(x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,255),2)
    cv2.imshow('result',img)

# 命名函数
def name():
    path = './saveimg/'
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = str(os.path.split(imagePath)[1].split('.',2)[1])
        names.append(name)

# 调用摄像头
cap = cv2.VideoCapture(0)
# 调用命名函数
name()
# 循环判断
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break

# 关闭所有窗口
cv2.destroyAllWindows()
# 结束摄像头调用
cap.release()

```

