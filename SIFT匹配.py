import cv2

# 读取图片
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 初始化SIFT检测器
sift = cv2.xfeatures2d.SIFT_create()

# 检测特征点
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 创建BFMatcher对象
bf = cv2.BFMatcher()

# 匹配特征点
matches = bf.knnMatch(des1, des2, k=2)

# 创建空列表
good = []

# 对匹配结果进行筛选
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# 计算图像的缩放比例
scale = 1.0 / (len(good) + 1.0)

# 计算匹配度
similarity = scale * len(good)

# 绘制匹配结果
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# 显示图像
cv2.imshow('SIFT Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 打印匹配度
print('Similarity: ', similarity)



#读取两幅图像、初始化SIFT检测器、检测特征点、创建BFMatcher对象、匹配特征点、对匹配结果进行筛选、计算图像的缩放比例、计算匹配度、绘制匹配结果以及显示图像等操作。
