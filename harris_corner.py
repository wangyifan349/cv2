import cv2
# 读取图片
img = cv2.imread('image.jpg')
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建哈里斯角检测器
harris_corner_detector = cv2.cornerHarris(gray, 3, 3, 0.04)
# 找出哈里斯角
harris_corners = cv2.goodFeaturesToTrack(harris_corner_detector, 50, 0.01, 10)
# 在图片中标记出哈里斯角
for corner in harris_corners:
    x, y = corner[0]
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
# 打印哈里斯角的坐标
for corner in harris_corners:
    x, y = corner[0]
    print("({}, {})".format(x, y))
# 显示图片
cv2.imshow('Image with Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
