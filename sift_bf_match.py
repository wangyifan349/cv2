def sift_bf_match(img1, img2, ratio=0.75):
    import cv2
    # 使用SIFT算子检测关键点和计算描述符
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 创建暴力匹配器对象
    bf = cv2.BFMatcher()

    # 对两个图像的描述符进行匹配
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 根据Lowe's ratio test筛选出好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # 返回筛选后的匹配结果
    return keypoints1, keypoints2, good_matches
  #keypoints1: 第一幅图像的关键点列表。
  #keypoints2: 第二幅图像的关键点列表。
  #good_matches: 经过筛选后的匹配结果列表，其中每个元素都是一个DMatch对象，它包含了两个关键点之间的距离、特征向量等信息。
  #关键点（KeyPoint）是图像中具有显著性、可以被描述子表示的局部特征点。在SIFT算法中，关键点通常是由高斯差分金字塔和尺度空间极值点检测得到的。
