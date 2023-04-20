import cv2

def orb_bf_match(img1, img2, ratio=0.75):
    # 使用ORB算子检测关键点和计算描述符
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 创建暴力匹配器对象
    bf = cv2.BFMatcher()

    # 对两个图像的描述符进行匹配
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    #descriptors1：第一幅图像的特征描述符，是一个numpy数组，大小为(n1, d)，其中n1表示关键点的数量，d表示每个特征向量的维度。
    #descriptors2：第二幅图像的特征描述符，大小为(n2, d)，其中n2表示关键点的数量，d表示每个特征向量的维度。
    #k=2：指定要匹配的最近邻数，即对于第一幅图像的每个关键点，在第二幅图像中找到两个最接近的关键点。这里我们将k设置为2，因为后面我们要使用Lowe's ratio test来筛选匹配结果。
    # 根据Lowe's ratio test筛选出好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # 返回筛选后的匹配结果
    return keypoints1, keypoints2, good_matches
#bf.knnMatch()函数的返回值是一个列表，其中每个元素都是每个查询关键点的最近邻匹配。具体地，对于第一幅图像中的每个关键点，该函数会在第二幅图像中找到k个最接近的关键点，并返回它们的距离和索引。在这里，我们只使用了距离最小和次小的两个最近邻匹配，以便后面进行Lowe's ratio test。
