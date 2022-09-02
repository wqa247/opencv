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
