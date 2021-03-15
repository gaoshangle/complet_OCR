import numpy as np
from PIL import Image
# from text.opencv_dnn_detect import angle_detect

import os


import cv2
import time

from math import *
from scipy.stats import mode


def detect_angle(img):
    """
    detect text angle in [0,90,180,270]
    @@img:np.array
    """
    angle = angle_detect(img)
    if angle == 90:
        im = Image.fromarray(img).transpose(Image.ROTATE_90)
        img = np.array(im)
    elif angle == 180:
        im = Image.fromarray(img).transpose(Image.ROTATE_180)
        img = np.array(im)
    elif angle == 270:
        im = Image.fromarray(img).transpose(Image.ROTATE_270)
        img = np.array(im)

    return img, angle

# 标准霍夫线变换
def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)   # theta是弧度
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
        y1 = int(y0 + 1000 * (a))   # 直线起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
        y2 = int(y0 - 1000 * (a))   # 直线终点纵坐标
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("image_lines", image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    start = time.time()
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    #                                                   表示成组成一条直线的最少点的数量，点数量不足的直线将被抛弃
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 300, minLineLength=100, maxLineGap=10) # 300 50 now the best (330,100,10)
    #                                                                      表示能被认为在一条直线上的亮点的最大距离
    print("dur:",time.time()-start)
    lines_corre = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        if x1 == x2 :# 水平
            result = 0
        elif y1 == y2:# 垂直
            result = 90
        else:# 角度
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result = np.arctan(k) * 57.2957795130823
        # print("直线倾斜角度为：" + str(result) + "度", int(x1), int(y1), int(x2), int(y2))
        if 0 <= np.abs(result) < 45:
            lines_corre.append(result)
    sta_angle = np.zeros((100),float)
    for i in range(0,len(lines_corre)):
        # ---处理成某整数的左右形式
        tem = lines_corre[i]
        if tem < 0:
            tem += 0.5 # whatever +/- should be add
        else :
            tem += 0.5
        #----
        x = tem + 45
        x = int (x)
        sta_angle[x] += 1
    max_value = max(sta_angle)
    indexs = [i for i,idx in enumerate(sta_angle) if idx == max_value]
    result_value = [] # 存放出现概率最高区间上的角度
    for i in indexs:
        tmp = i -45
        for j in lines_corre:
            if tmp - 0.5 <= j < tmp + 0.5:
                result_value.append(j)
    if len(result_value)==0:
        # cv2.imwrite('/home/cciip/桌面/111.jpg', image)
        return 0
    # 在概率分布的基础上求均值
    angle_mean = np.mean(result_value)
    # cv2.imwrite('/home/cciip/桌面/111.jpg', image) # now image is red thread img
    return angle_mean


def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    img = cv2.warpAffine(image, M, (w, h))
    return img

def angle_corre(path):
    img = cv2.imread(path)
    angle = line_detect_possible_demo(img)
    print('angle:',angle)
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = cv_img.copy()
    img = rotate_bound(img, -float(angle))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print('/home/share/gaoluoluo/dbnet/test/test_corre_input/'+path.split('/')[-1])
    cv2.imwrite('./invoices_gao_true/corre_gao/'+path.split('/')[-1], img)
    return img

# '/home/share/gaoluoluo/dbnet/test/test_corre_input/0.jpg'

# def rotate_bound(image, angle):
#     # 获取宽高
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
#     img = cv2.warpAffine(image, M, (w, h))
#     return img
#
#
# def rotate_points(points, angle, cX, cY):
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0).astype(np.float16)
#     a = M[:, :2]
#     b = M[:, 2:]
#     b = np.reshape(b, newshape=(1, 2))
#     a = np.transpose(a)
#     points = np.dot(points, a) + b
#     points = points.astype(np.int)
#     return points
#
#
# def findangle(_image):
#     # 用来寻找当前图片文本的旋转角度 在±90度之间
#     # toWidth: 特征图大小：越小越快 但是效果会变差
#     # minCenterDistance：每个连通区域坐上右下点的索引坐标与其质心的距离阈值 大于该阈值的区域被置0
#     # angleThres：遍历角度 [-angleThres~angleThres]
#
#     toWidth = _image.shape[1] // 2  # 500
#     minCenterDistance = toWidth / 20  # 10
#     angleThres = 45
#
#     image = _image.copy()
#     h, w = image.shape[0:2]
#     if w > h:
#         maskW = toWidth
#         maskH = int(toWidth / w * h)
#     else:
#         maskH = toWidth
#         maskW = int(toWidth / h * w)
#     # 使用黑色填充图片区域
#     swapImage = cv2.resize(image, (maskW, maskH))
#     # print(swapImage)
#     # print("---------------------")
#     grayImage = cv2.cvtColor(swapImage, cv2.COLOR_BGR2GRAY)
#     # print(grayImage)
#     gaussianBlurImage = cv2.GaussianBlur(grayImage, (3, 3), 0, 0)
#     histImage = cv2.equalizeHist(~gaussianBlurImage)
#     binaryImage = cv2.adaptiveThreshold(histImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
#     # pointsNum: 遍历角度时计算的关键点数量 越多越慢 建议[5000,50000]之中
#     pointsNum = np.sum(binaryImage != 0) // 2
#
#     # # 使用最小外接矩形返回的角度作为旋转角度
#     # # >>一步到位 不用遍历
#     # # >>如果输入的图像切割不好 很容易受干扰返回0度
#     # element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     # dilated = cv2.dilate(binaryImage*255, element)
#     # dilated = np.pad(dilated,((50,50),(50,50)),mode='constant')
#     # cv2('dilated', dilated)
#     # coords = np.column_stack(np.where(dilated > 0))
#     # angle = cv2.minAreaRect(coords)
#     # print(angle)
#
#     # 使用连接组件寻找并删除边框线条
#     # >>速度比霍夫变换快5~10倍 25ms左右
#     # >>计算每个连通区域坐上右下点的索引坐标与其质心的距离，距离大的即为线条
#     connectivity = 8
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage, connectivity, cv2.CV_8U)
#     # print(num_labels)
#     # print("-----------------------")
#     # print(labels)
#     # print("------------------------")
#     # print(stats)
#     # print("------------------------")
#     # print(centroids)
#     # print("--------------------------")
#
#     labels = np.array(labels)
#     maxnum = [(i, stats[i][-1], centroids[i]) for i in range(len(stats))]
#     maxnum = sorted(maxnum, key=lambda s: s[1], reverse=True)
#     if len(maxnum) <= 1:
#         return 0
#     for i, (label, count, centroid) in enumerate(maxnum[1:]):
#         cood = np.array(np.where(labels == label))
#         distance1 = np.linalg.norm(cood[:, 0] - centroid[::-1])
#         distance2 = np.linalg.norm(cood[:, -1] - centroid[::-1])
#         if distance1 > minCenterDistance or distance2 > minCenterDistance:
#             binaryImage[labels == label] = 0
#         else:
#             break
#
#
#     minRotate = 0
#     minCount = -1
#     (cX, cY) = (maskW // 2, maskH // 2)
#     points = np.column_stack(np.where(binaryImage > 0))[:pointsNum].astype(np.int16)
#     print("points:",np.shape(points)) # (208986, 2)
#     for rotate in range(-angleThres, angleThres):
#         rotatePoints = rotate_points(points, rotate, cX, cY)
#         rotatePoints = np.clip(rotatePoints[:, 0], 0, maskH - 1)
#         hist, bins = np.histogram(rotatePoints, maskH, [0, maskH])
#         # 横向统计非零元素个数 越少则说明姿态越正
#         zeroCount = np.sum(hist > toWidth / 50)
#         if zeroCount <= minCount or minCount == -1:
#             minCount = zeroCount
#             minRotate = rotate
#
#     # print("over: rotate = ", minRotate)
#     return minRotate
#
# def get_rotate_img(path):
#
#
#     # assert os.path.exists(path), 'file is not exists'
#     # img = cv2.imread(path, 1 if self.img_mode != 'GRAY' else 0)
#
#
#     cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#     # print("cv_img")
#     # print(cv_img)
#     cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
#
#
#     img = cv_img.copy()
#     # print("img")
#     # print(img)
#     angle = findangle(img)
#     # print(cv2.__version__)
#     print("angle:",angle)
#     img = rotate_bound(img, -angle)
#     # misc.imsave(Path,img)
#     # img = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2RGB) # 转换为RGB
#     # print("img")
#     # print(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img
#
#
# # 角度旋转
#
# class ImgCorrect() :
#     def __init__(self, img) :
#         self.img = img
#         self.h, self.w, self.channel = self.img.shape
#         if self.w <= self.h :
#             self.scale = 2000 / self.w
#             self.w_scale = 2000
#             self.h_scale = self.h * self.scale
#             self.img = cv2.resize(self.img, (0, 0), fx = self.scale, fy = self.scale, interpolation = cv2.INTER_NEAREST)
#         else :
#             self.scale = 1190 / self.h
#             self.h_scale = 1190
#             self.w_scale = self.w * self.scale
#             self.img = cv2.resize(self.img, (0, 0), fx = self.scale, fy = self.scale, interpolation = cv2.INTER_NEAREST)
#         self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
#
#     def img_lines(self) :
#         ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#         # cv2.imshow("bin",binary)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
#         binary = cv2.dilate(binary, kernel)  # 膨胀
#         edges = cv2.Canny(binary, 50, 200)
#         # cv2.imshow("edges", edges)
#         self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength = 100, maxLineGap = 20)
#         # print(self.lines)
#         if self.lines is None :
#             print("Line segment not found")
#             return None
#
#         lines1 = self.lines[:, 0, :]  # 提取为二维
#         # print(lines1)
#         imglines = self.img.copy()
#         for x1, y1, x2, y2 in lines1[:] :
#             cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         return imglines
#
#     def search_lines(self) :
#         lines = self.lines[:, 0, :]  # 提取为二维
#         # k = [(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in lines]
#         # sorted_k = sorted(lines, key=lambda x:(x[3] - x[1]) / (x[2] - x[0]))
#         number_inexistence_k = 0
#         sum_positive_k45 = 0
#         number_positive_k45 = 0
#         sum_positive_k90 = 0
#         number_positive_k90 = 0
#         sum_negative_k45 = 0
#         number_negative_k45 = 0
#         sum_negative_k90 = 0
#         number_negative_k90 = 0
#         sum_zero_k = 0
#         number_zero_k = 0
#         for x in lines :
#             if x[2] == x[0] :
#                 number_inexistence_k += 1
#                 continue
#             # print(degrees(atan((x[3] - x[1]) / (x[2] - x[0]))), "pos:", x[0], x[1], x[2], x[3], "斜率:",
#             #       (x[3] - x[1]) / (x[2] - x[0]))
#             if 0 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 45 :
#                 number_positive_k45 += 1
#                 sum_positive_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
#             if 45 <= degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 90 :
#                 number_positive_k90 += 1
#                 sum_positive_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
#             if -45 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) < 0 :
#                 number_negative_k45 += 1
#                 sum_negative_k45 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
#             if -90 < degrees(atan((x[3] - x[1]) / (x[2] - x[0]))) <= -45 :
#                 number_negative_k90 += 1
#                 sum_negative_k90 += degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
#             if x[3] == x[1] :
#                 number_zero_k += 1
#
#         max_number = max(number_inexistence_k, number_positive_k45, number_positive_k90, number_negative_k45,
#                          number_negative_k90, number_zero_k)
#         # print(number_inexistence_k,number_positive_k45, number_positive_k90, number_negative_k45, number_negative_k90,number_zero_k)
#         if max_number == number_inexistence_k :
#             return 90
#         if max_number == number_positive_k45 :
#             return sum_positive_k45 / number_positive_k45
#         if max_number == number_positive_k90 :
#             return sum_positive_k90 / number_positive_k90
#         if max_number == number_negative_k45 :
#             return sum_negative_k45 / number_negative_k45
#         if max_number == number_negative_k90 :
#             return sum_negative_k90 / number_negative_k90
#         if max_number == number_zero_k :
#             return 0
#
#     def rotate_image(self, degree) :
#         """
#         正角 逆时针旋转
#         :param degree:
#         :return:
#         """
#         print("degree:", degree)
#         if -45 <= degree <= 0 :
#             degree = degree  # #负角度 顺时针
#         if -90 <= degree < -45 :
#             degree = 90 + degree  # 正角度 逆时针
#         if 0 < degree <= 45 :
#             degree = degree  # 正角度 逆时针
#         if 45 < degree < 90 :
#             degree = degree - 90  # 负角度 顺时针
#         print("rotate degree:", degree)
#         # degree = -45
#         # # 获取旋转后4角的填充色
#         filled_color = -1
#         if filled_color == -1 :
#             filled_color = mode([self.img[0, 0], self.img[0, -1],
#                                  self.img[-1, 0], self.img[-1, -1]]).mode[0]
#         if np.array(filled_color).shape[0] == 2 :
#             if isinstance(filled_color, int) :
#                 filled_color = (filled_color, filled_color, filled_color)
#         else :
#             filled_color = tuple([int(i) for i in filled_color])
#
#         # degree = degree - 90
#         height, width = self.img.shape[:2]
#         heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
#         widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
#
#         matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # 逆时针旋转 degree
#
#         matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
#         matRotation[1, 2] += (heightNew - height) / 2
#
#         imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue = filled_color)
#
#         return imgRotation
#
#
# def angle_correct(img_path) :
#     im = cv2.imread(img_path)
#     imgcorrect = ImgCorrect(im)
#     # cv2.imshow('ormalization image', mgcorrect.img)
#     lines_img = imgcorrect.img_lines()
#     # print(type(lines_img))
#     if lines_img is None :
#         imgcorrect = imgcorrect.rotate_image(0)
#     # cv2.imshow("lines_img",lines_img)
#     else :
#         # cv2.imshow("lines_img", lines_img)
#         degree = imgcorrect.search_lines()
#         # degree = degree * 0.5
#         imgcorrect = imgcorrect.rotate_image(degree)
#     img_name = str(img_path).split('/')[-1]
#     new_path = '/home/share/gaoluoluo/dbnet/test/test_corre_input/'+img_name
#     cv2.imwrite(new_path, imgcorrect)
#     return new_path
#     # cv2.waitKey()
