 #param img1: 第一张图片
 #param img2: 第二张图片
import cv2
def SIFT_match(img1, img2):

    # 导入SIFT模块
    from skimage.feature import SIFT
    # 创建SIFT对象
    sift = SIFT()
    # 计算图像的SIFT特征
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 计算特征点的匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 对匹配结果进行筛选，筛选出距离比率小于0.75的匹配结果
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # 计算图像的缩放比例
    img1_scale = img1.shape[0] / img2.shape[0]
    # 计算匹配结果
    result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # 将结果中的特征点的坐标进行缩放
    for pt1, pt2 in good:
        pt1.pt = (pt1.pt[0] / img1_scale, pt1.pt[1] / img1_scale)
    # 返回匹配结果
    return result
