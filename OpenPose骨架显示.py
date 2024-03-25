import cv2
import numpy as np
from openpose import pyopenpose as op

# 初始化 OpenPose
params = {
    "model_folder": "path_to_openpose_models",
    "hand": True,
    "face": True
}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧大小以适应 OpenPose
    frame = cv2.resize(frame, (640, 480))

    # OpenPose 处理帧
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # 提取关键点
    keypoint_list = datum.poseKeypoints
    face_keypoint_list = datum.faceKeypoints
    hand_keypoint_list = datum.handKeypoints

    # 在帧上绘制关键点
    if keypoint_list is not None:
        for person in keypoint_list:
            for keypoint in person:
                if keypoint[2] > 0.1:  # 可信度阈值
                    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

    if face_keypoint_list is not None:
        for face in face_keypoint_list:
            for keypoint in face:
                if keypoint[2] > 0.1:  # 可信度阈值
                    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

    if hand_keypoint_list is not None:
        for hand in hand_keypoint_list:
            for keypoint in hand:
                if keypoint[2] > 0.1:  # 可信度阈值
                    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # 显示帧
    cv2.imshow("Frame", frame)

    # 检测按键并退出
    key = cv2.waitKey(1)
    if key == 27:  # ESC键退出
        break

# 清理
cap.release()
cv2.destroyAllWindows()
