import cv2
# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')
# 打开摄像头
cap = cv2.VideoCapture(0)
# 定义计数器和帧率
count = 0
frame_rate = 5
while True:
    ret, frame = cap.read()
    # 如果读取失败，则退出循环
    if not ret:
        break
    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # 对每个检测到的人脸进行标注
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 每5帧保存一张图片
    if count % frame_rate == 0:
        cv2.imwrite(f'frame_{count//frame_rate}.jpg', frame)
    cv2.imshow('frame', frame)

    # 按下q键退出循环
    if cv2.waitKey(1) == ord('q'):
        break
    count += 1

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
