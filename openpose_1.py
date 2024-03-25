import cv2
import pyopenpose as op

# 设置参数
params = {
    "model_folder": "models/",
    "face": True,
    "hand": True
}

# 启动 OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 读取图像
image_path = "image.jpg"
image = cv2.imread(image_path)

# 创建 Datum 对象，并将图片传入
datum = op.Datum()
datum.cvInputData = image

# 处理图像
opWrapper.emplaceAndPop([datum])

# 获取关键点
keypoints = datum.poseKeypoints
face_keypoints = datum.faceKeypoints
hand_keypoints = datum.handKeypoints

# 在图像上绘制关键点
output_image = datum.cvOutputData

# 显示图像
cv2.imshow("OpenPose", output_image)
cv2.waitKey(0)

# 保存结果
cv2.imwrite("output_image.jpg", output_image)

cv2.destroyAllWindows()
