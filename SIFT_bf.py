import cv2
import numpy as np

# 读取图片
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 将图片转换为灰度图
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 检测关键点
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 创建Brute-Force Matcher对象
bf = cv2.BFMatcher()

# 根据描述符匹配关键点
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 计算比率
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# 绘制匹配结果
img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

# 计算图片的尺寸和缩放比例
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 计算图片的缩放比例
scale = max(h1/h2, w1/w2)

# 调整图片大小
img2 = cv2.resize(img2, (int(w2*scale), int(h2*scale)))

# 将图片合并
img3 = np.hstack((img1, img2))

# 显示结果
cv2.imshow('Result', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
