import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 加载查询图像（要搜索的图像）
img1 = cv.imread('box.png', 0)
img2 = cv.imread('box_in_scene.png', 0)

# 使用cv.SIFT_create()方法初始化SIFT检测器
sift = cv.SIFT_create()

# 使用SIFT在查询图像中找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)

# 使用SIFT在训练图像中找到关键点和描述符
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN（快速最近邻居搜索库）参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)# 或者传递一个空字典

# 创建FLANN匹配器
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)# 使用匹配器找到每个查询图像描述符的k个最佳匹配


matchesMask = [[0, 0] for _ in range(len(matches))]# 只需要绘制好的匹配项，因此创建一个掩码

for i, (m, n) in enumerate(matches):# 根据Lowe的论文进行比率测试
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]


draw_params = dict(matchColor=(0, 255, 0),# 用于显示匹配项的绘图参数
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)# 绘制并显示匹配项


plt.imshow(img3)# 使用matplotlib显示图像
plt.show()
