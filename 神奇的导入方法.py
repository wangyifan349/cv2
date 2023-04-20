import wget

# 下载文件到本地
url = 'https://raw.githubusercontent.com/wangyifan349/cv2/main/orb_bf_match.py'#一个简单的匹配方法在我的github开放仓库
filename = wget.download(url)

# 导入函数
from orb_bf_match import orb_bf_match

# 调用函数
img1 = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')
kp1, kp2, matches = orb_bf_match(img1, img2)
#好玩的办法，以后代码全部从云端导入了，本地没代码了?!!!啊哈哈哈

