# -*- coding:utf-8 -*-
# import os
# import cv2
# import sys
# import time
import math
import copy
from functools import reduce
# import collections
# import torch
import argparse
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy import misc, ndimage
# from angle import get_rotate_img

from PIL import Image

# from torch.autograd import Variable
# from torch.utils import data
#
# from dataset import DataLoader
# import models
import util
# c++ version pse based on opencv 3+
# from pse import pse

# from angle import detect_angle
from apphelper.image import order_point, calc_iou, xy_rotate_box
from crnn import crnnRec

# from eval.invoice.eval_invoice import evaluate
# from layout.VGGLocalization import VGGLoc, trans_image

# python pse
# from pypse import pse as pypse
# from pse2 import pse2
# from angle import *

def get_right_marge(data):
    # 排除备注中的信息，以图片的中间为限
    max_r = 0
    for dd in data:
        tem_r = float(dd['cx']) + float(dd['w']) / 2
        if tem_r > max_r:
            max_r = tem_r
    return max_r + 200


def get_min_distance_index(tem,indexes,near):
    x = float(tem['cx']) + float(tem['w']) / 2
    y = float(tem['cy']) + float(tem['h']) / 2
    distance_min = 100000
    ii = 0
    for i in indexes:
        x_tem = float(near[i]['cx']) - float(near[i]['w']/2)
        y_tem = float(near[i]['cy']) + float(near[i]['h']/2)
        distance_tem = math.sqrt(math.pow((x-x_tem),2) + math.pow((y-y_tem),2))
        if distance_tem < distance_min and len(near[i]['text']) and calc_axis_iou(near[i],tem,1) > 0.3:
            ii = i
            distance_min = distance_tem
    return ii

def recognize(im=None, path=None):
    ret = None
    try:
        file = None
        if im:
            pass
        elif path:
            im = Image.open(path)
            file = os.path.basename(path)

        if im:
            dir = '/home/share/gaoluoluo/complete_ocr/data/images/tmp/'
            file = file if file is not None else 'tmp.jpg'
            tmpfile = os.path.join(dir, file)
            im.save(tmpfile)

            data = test(None, tmpfile)
            if data:
                ret = format(data)
    except Exception as e:
        print(e)

    return ret


def format(data):
    return data


def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img


def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)


def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)

#----
# --
# -----------------------------------------------------------------------------------

import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

# project = 'DBNet.pytorch'  # 工作项目根目录
# sys.path.append(os.getcwd().split(project)[0] + project)
import time
import cv2
import torch

from dbnet.data_loader import get_transforms
from dbnet.models import build_model
from dbnet.post_processing import get_post_processing

def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件) model_path='/home/share/gaoluoluo/dbnet/output/DBNet_resnet18_FPN_DBHead/checkpoint/model_latest.pth'
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img, is_output_polygon=False, short_size: int = 1024):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2] # 2550 3507
        img = resize_image(img, short_size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t

def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    # parser.add_argument('--model_path', default=r'/home/share/gaoluoluo/dbnet/output/DBNet_resnet50_FPN_DBHead/checkpoint/model_latest.pth', type=str)
    # parser.add_argument('--model_path', default=r'/home/share/gaoluoluo/模型/model50_ch_epoch510_latest.pth', type=str)
    parser.add_argument('--model_path', default=r'/home/share/gaoluoluo/模型/model50_ch_epoch248_latest.pth', type=str)
    parser.add_argument('--input_folder', default='/home/share/gaoluoluo/complete_ocr/data/img_not_use', type=str, help='img path for predict')
    # parser.add_argument('--input_folder', default='/home/share/gaoluoluo/dbnet/test/test_input', type=str, help='img path for predict')
    parser.add_argument('--img_correct',default='/home/share/gaoluoluo/dbnet/test/test_corre_input/',type=str,help='img_correct path for predict')
    # parser.add_argument('--input_folder',default='/home/share/gaoluoluo/dbnet/test/test_corre_input', type=str, help='img path for predict')
    # parser.add_argument('--input_folder', default='/home/share/gaoluoluo/complete_ocr/data/images/tmp', type=str,help='img path for predict')
    # parser.add_argument('--output_folder', default='/home/share/gaoluoluo/dbnet/test/test_output', type=str, help='img path for output')
    parser.add_argument('--output_folder', default='/home/share/gaoluoluo/dbnet/test/test_output', type=str, help='img path for output')
    parser.add_argument('--gt_txt', default='/home/share/gaoluoluo/complete_ocr/data/txt_not_use', type=str, help='img 对应的 txt')
    parser.add_argument('--thre', default=0.1, type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_result', action='store_true', help='save box and score to txt file')
    parser.add_argument('--evaluate', nargs='?', type=bool, default=True,
                        help='evalution')
    args = parser.parse_args()
    return args

def test(args, file=None):

    import pathlib
    import matplotlib.pyplot as plt
    from dbnet.utils.util import show_img, draw_bbox, save_result, get_file_list
    args = init_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    # 初始化网络                                         0.1
    model = Pytorch_model(args.model_path, post_p_thre=args.thre, gpu_id=0)
    total_frame = 0.0
    total_time = []
    names = []
    for ss in range(0,1): # only one picture app detect
        img_path = file
        print("\nimg_path:", img_path)
        start_time = time.time()
        img = angle_corre(img_path)# 调整图片角度
        img_path = args.img_correct + img_path.split('/')[-1] # After correct angle img path
        names.append(img_path.split('/')[-1])
        print("img_path:",img_path)
        preds, boxes_list, score_list, t = model.predict(img, is_output_polygon=args.polygon)
        img_path1 = img_path
        box = []
        for i in range(0,len(boxes_list)):
            for  j in range(0,len(boxes_list[0])):
                b = []
                b.append(np.float32(boxes_list[i][j][0]))
                b.append(np.float32(boxes_list[i][j][1]))
                box.append(b)
        boxes_list = box
        #合框只能处理一个框被分成两个部分
        i = 4
        points_kuang = []
        remove_mark = []
        max_X = -1
        while(i<=len(boxes_list)):
            points = boxes_list[i-4:i]
            for _,p in enumerate(points):
                if p[0] > max_X:
                    max_X = p[0]
            i = i+4
            points = np.array(points)
            points_kuang.append(points)
        for i in range(10,len(points_kuang)-10):
            point3 = points_kuang[i][2]
            start_point = i - 8
            end_point = i + 8
            if start_point < 0:
                start_point = 0
            if end_point > len(points_kuang):
                end_point = len(points_kuang)
            if i not in remove_mark:
                for j in range(start_point,end_point):
                    point4 = points_kuang[j][3]
                    min_dis = math.sqrt(math.pow((point3[0] - point4[0]),2) + math.pow((point3[1] - point4[1]),2))
                    Y_cha = math.fabs(point3[1] - point4[1])
                    if min_dis < 15 and point4[0] > max_X / 2 and Y_cha < 25 and j not in remove_mark and i != j: # 10 reasonable
                        point1_1 = points_kuang[i][0]
                        point1_2 = points_kuang[j][0]
                        x_min = min(point1_1[0],point1_2[0])
                        y_min = min(point1_1[1],point1_2[1])
                        point3_2 = points_kuang[j][2]
                        x_max = max(point3[0],point3_2[0])
                        y_max = max(point3[1],point3_2[1])
                        points_kuang[i][0,0] = x_min
                        points_kuang[i][0,1] = y_min
                        points_kuang[i][1,0] = x_max
                        points_kuang[i][1,1] = y_min
                        points_kuang[i][2,0] = x_max
                        points_kuang[i][2,1] = y_max
                        points_kuang[i][3,0] = x_min
                        points_kuang[i][3,1] = y_max
                        remove_mark.append(j)
                        break
        remove_mark = sorted(remove_mark,reverse=True)
        for _,i in enumerate(remove_mark):
            del points_kuang[i]
        boxes_list_save = points_kuang # 决定保存的画框图片是否是 合框之后的图片
        #---
        i = 0;
        rects = []
        while(i<len(points_kuang)):
            points = points_kuang[i]
            rect = cv2.minAreaRect(points) # 4个点 -> d cx cy w h
            rec = []
            rec.append(rect[-1])
            rec.append(rect[1][1])
            rec.append(rect[1][0])
            rec.append(rect[0][0])
            rec.append(rect[0][1])
            rects.append(rec)
            i += 1
        ori_img = cv2.imread(img_path1)
        result = crnnRec(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB), rects)
        result = list(reversed(result))
        dur = time.time() - start_time
        print("dur:",dur)
        #  保存画框图片---------
        # img = draw_bbox(img(img_path)[:, :, ::-1], boxes_list_save)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list_save)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')  # /home/share/gaoluoluo/dbnet/test/output/2018实验仪器发票_result.jpg
        pred_path = os.path.join(args.output_folder,img_path.stem + '_pred.jpg')  # /home/share/gaoluoluo/dbnet/test/output/2018实验仪器发票_pred.jpg
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list_save, score_list, args.polygon)
        #  --------
        total_frame += 1
        total_time.append(dur)
        try:
            result = formatResult(result)
        except Exception as e:
            print(e)
            continue
    return result

def predict_bbox(args, model, org_img, img, slice):
    # angle detection
    # org_img, angle = detect_angle(org_img)
    outputs = model(img)

    score = torch.sigmoid(outputs[:, slice, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

    text = outputs[:, slice, :, :]
    kernels = outputs
    # kernels = outputs[:, 0:args.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

    if args.arch == 'mobilenet':
        pred = pse2(kernels, args.min_kernel_area / (args.scale * args.scale))
    else:
        # c++ version pse
        pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
    scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []
    rects = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < args.min_area / (args.scale * args.scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * scale
        bbox = bbox.astype('int32')
        bbox = order_point(bbox)
        # bbox = np.array([bbox[1], bbox[2], bbox[3], bbox[0]])
        bboxes.append(bbox.reshape(-1))

        rec = []
        rec.append(rect[-1])
        rec.append(rect[1][1] * scale[1])
        rec.append(rect[1][0] * scale[0])
        rec.append(rect[0][0] * scale[0])
        rec.append(rect[0][1] * scale[1])
        rects.append(rec)

    return bboxes, rects, text

import re
def isTitle(text):

    return re.search(r'(货物|劳务|服.?名称|规.?型号|^单[位价]|^数量|税率|税额)', text) != None
    return re.search(r'(货物|劳务|服.?名称|规.?型号|^单[位价]|^数量|.?额$|税率|税额|项目|类型|车牌|通行日期)', text) != None

def isSummary(text):
    pattern = re.compile(r'(￥|Y|羊)[0-9]+(.[0-9]{2})$')
    return text==r'计' or text == r'合' or text == r'合计' or pattern.search(text) != None

def get_content_boundary(data):
    #--- 过滤左边的无用信息
    left = [d for i, d in enumerate(data) if isTitle(d['text'])]
    left = sorted(left,key=lambda x: x['cx'] - x['w'] / 2)
    left = left[0]
    cx_tem = float(left['cx'] - float(left['w']) / 2) # 标题的左
    if re.search(r'(货物|劳务|服.?名称)', left['text']) != None:
        for i in range(len(data)-1,-1,-1):
            cx = float(data[i]['cx']) + float(data[i]['w']) / 2 # 其他的右
            if(0 < cx_tem -cx):
                # print(data[i])
                data.pop(i)

    while len(data):
        left_tem = min(data, key = lambda x : x['cx'] - x['w'] / 2)
        if len(left_tem['text']) < 3 :
            data.remove(left_tem)
        else :
            break
    while len(data):
        right_tem = max(data, key = lambda x : x['cx'] + x['w'] / 2)
        if len(right_tem['text']) < 3 :
            data.remove(right_tem)
        else :
            break
    #----
    title = [i for i, d in enumerate(data) if isTitle(d['text'])]
    s = title[0]
    e = title[-1]
    title.extend([i+s-4 for i,d in enumerate(data[s-4:s]) if abs(d['cy'] - data[s]['cy']) < 10])
    title.extend([i+e+1 for i,d in enumerate(data[e+1:e+6]) if abs(d['cy'] - data[e]['cy']) < 10])
    s = min(title)
    e = max(title)
    lf = min(data[s:e+1], key=lambda x:x['cx']-x['w']/2)

    summary = [i for i, d in enumerate(data) if isSummary(d['text'])]
    s = summary[0]
    summary.extend([i+s-4 for i, d in enumerate(data[s-4:s]) if abs(d['cy'] - data[summary[-1]]['cy']) < 10])
    s = min(summary)

    start = e
    end = s
    rt = max(data[start:end], key=lambda x: x['cx']+x['w']/2)


    left = lf['cx']-lf['w']/2   # 20为经验值
    right = rt['cx']+rt['w']/2   # 80 为经验值

    return (start, end, left, right)

def check(data, placeholder, dir = 0):
    try:
        i = None
        if isinstance(placeholder,list):
            for pld in placeholder:
                f = [x for x,d in enumerate(data) if d == pld]
                if len(f):
                    i = f[-1]
                    break
        else:
            f = [x for x,d in enumerate(data) if d == placeholder]
            if len(f):
                i = f[-1]
        return i
    except:
        return None

LEFT_MARGIN = 70
RIGHT_MARGIN = 20
def parseLine(line, boundary, isvehicle=False):
    xmin, xmax = boundary

    copyed = copy.deepcopy(line)
    copyed = preprocess_line(copyed) # 主要处理的是框框交叉,并对cx排序

    if isvehicle:
        # ratio, price = get_ratio(copyed, xmax)
        ratio, price, tax = get_ratio_price_tax(copyed, xmax)
        # title
        title = get_title(copyed, xmin)
        sdate, edate, price, ratio = get_date(copyed, price, ratio)
        platenum, cartype, sdate, edate = get_platenum_cartype(copyed, sdate, edate)

        return postprocess_line(title, platenum, cartype, sdate, edate, price, ratio, tax)
    else:

        ratio, price, tax = get_ratio_price_tax(copyed, xmax)
        # title
        title = get_title(copyed, xmin) # jin
        # tax
        # tax = get_tax(copyed, xmax, ratio)
        # prices
        #
        specs, amount, uprice, price, ratio = get_numbers(copyed, price, ratio,tax)
        #specs
        specs,unit = get_specs_unit(copyed, specs)

        return postprocess_line(title, specs, unit, amount, uprice, price, ratio, tax)

def preprocess_line(line):
    line = sorted(line, key=lambda x: x['cx'] - x['w'] / 2)
    res = []

    i = 0
    j = 1
    while i < len(line) and j < len(line):
        x = line[i]
        y = line[j]
        x1 = x['cx'] + x['w'] / 2 # 左的右边
        y1 = y['cx'] - y['w'] / 2 # 右的左边

        if abs(x1 - y1) < 8: # 说明是同一个框 合并
            x['w'] = y['cx'] + y['w']/2 - x['cx'] + x['w']/2
            x['cx'] = y['cx'] + y['w']/2 - x['w']/2
            x['text'] = x['text'] + y['text']
            j = j + 1
        else: # 不是同一个框，继续向you遍历
            res.append(x)
            i = j
            j = j + 1

    res.append(line[i])
    return res

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def postprocess_line(title, specs, unit, amount, uprice, price, ratio, tax):
    if tax != '***' and is_number(price) and ratio and (not tax or not is_number(tax) or float(tax) > float(price) or not '.' in tax) :
        tax = '%.2f' % (float(price) / 100 * float(ratio[:-1]))
    return [title, specs, unit, amount, uprice, price, ratio, tax]

def get_title(line, xmin):
    title = ''
    candidates = [d for d in line if abs(d['cx'] - d['w'] / 2 - xmin) < LEFT_MARGIN]
    if len(candidates):
        candidates = sorted(candidates, key=lambda x:x['cy'])
        for c in candidates:
            title += c['text']
            line.remove(c)

    if title and not title.startswith('*'):
        title = '*' + title

    return title

def get_ratio_price_tax(line, xmax):
    ratio = ''
    price = ''
    tax = ''

    pat = re.compile(r'(\-?[\dBG]+\.?[\dBG]{2})*(([\dBG]+[\%])|([\u4e00-\u9fa5]+税))$')
    pat2 = re.compile(r'(\-?[\dBG]+\.[\dBG]{2})([\dBG]{1,2}[8])')

    ratioitem = None
    for i in range(len(line)-1, -1, -1):
        text = line[i]['text']
        m = pat.match(text)
        if m:
            price, ratio = (m.group(1), m.group(2)) if m.group(1) else ('', m.group(2))
        else:
            if len(line) - i < 3: # reduce
                m = pat2.match(text)
                if m:
                    price, ratio = (m.group(1), m.group(2))
                    ratio = ratio[:-1] + '%'

        if ratio:
            ratioitem = line.pop(i)
            break

    if not ratio:
        numbers = sorted([i for i,d in enumerate(line) if re.match(r'([\dBG]+\.?[\dBG]{2})*', d['text'])], key=lambda x:line[x]['cx']-line[x]['w']/2)
        if len(numbers)>=3:
            i = numbers[-2]
            d = line[i]
            m = re.match(r'(\d{1,2})\D+', d['text']) # 1-2 number and !number char eg 11%
            if m:
                ratio = m.group(1)
                ratioitem = line.pop(i)

    if re.search(r'税$', ratio):
        tax = '***'
    else:
        if ratioitem:
            taxes = [l for l in line if l['cx'] > ratioitem['cx']]
            if len(taxes):
                tax = taxes[0]['text']
                line.remove(taxes[0])

        if not tax:
            x = 1000
            for i,d in enumerate(line):
                if abs(d['cx'] + d['w'] / 2 - xmax) < x:
                    x = abs(d['cx'] + d['w'] / 2 - xmax)
            idx = [i for i, d in enumerate(line) if abs(d['cx'] + d['w'] / 2 - xmax) < RIGHT_MARGIN]
            if len(idx):
                idx = idx[0]
                ln = line[idx]
                tax = ln['text']
                line.pop(idx)

                if len(tax) > 2 and tax.find('.') == -1:
                    tax = tax[:-2] + '.' + tax[-2:]

        if len(price) and not '.' in price:
            if tax and ratio:
                while float(price) > float(tax):
                    prc = price[:-2] + '.' + price[-2:]
                    f_tax = float(tax)
                    f_ratio = float(ratio[:-1])
                    f_price = float(prc)

                    if abs(f_price * f_ratio / 100.0 - f_tax) > 0.1 and f_ratio < 10:
                        ratio = price[-1] + ratio
                        price = price[:-1]
                    else:
                        price = prc
                        break
            else:
                price = price[:-2] + '.' + price[-2:]

    if price and ratio and not tax:
        tax = str(round(float(price) * float(ratio[:-1]) / 100.0, 2))

    return ratio, price, tax

def get_tax(line, xmax, ratio):
    tax = ''

    if re.search(r'税$', ratio):
        tax = '***'

    idx = [i for i, d in enumerate(line) if abs(d['cx'] + d['w'] / 2 - xmax) < RIGHT_MARGIN]
    if len(idx):
        idx = idx[0]
        ln = line[idx]
        tax = ln['text']
        line.pop(idx)

        if len(tax) > 2 and tax.find('.') == -1:
            tax = tax[:-2] + '.' + tax[-2:]

    return tax

def get_ratio(line, xmax):
    ratio = ''
    price = ''
    pat = re.compile(r'(\-?[\dBG]+\.?[\dBG]{2})*(([\dBG]+[\%])|([\u4e00-\u9fa5]+税))$')

    for i in range(len(line)-1, -1, -1):
        text = line[i]['text']
        m = pat.match(text)
        if m:
            price, ratio = (m.group(1), m.group(2)) if m.group(1) else ('', m.group(2))

        if ratio:
            line.pop(i)
            break

    if not ratio:
        numbers = sorted([i for i,d in enumerate(line) if re.match(r'([\dBG]+\.?[\dBG]{2})*', d['text'])], key=lambda x:line[x]['cx']-line[x]['w']/2)
        if len(numbers)>=3:
            i = numbers[-2]
            d = line[i]
            m = re.match(r'(\d{1,2})\D+', d['text'])
            if m:
                ratio = m.group(1) + '%'
                line.pop(i)

    return ratio, price

def get_numbers(line, price, ratio,tax):

    specs = ''
    amount = ''
    uprice = ''

    pattern = re.compile(r'\-?[\dBG:]+\.?[\dBG]*$')
    numbers = []

    for d in line:
        if pattern.match(d['text']):
            d['text'] = d['text'].replace('B','8').replace('G', '6').replace(':','')
            numbers.append(d)

    if len(numbers):
        for n in numbers:
            line.remove(n)

        # preprocess_number(numbers)
        numbers = sorted(numbers, key=lambda x: x['cx'] - x['w'] / 2)
        if not ratio and re.match(r'^\d{2,3}$', numbers[-1]['text']):
            ratio = numbers[-1]['text']
            ratio = ratio[:-1] + '%'
            numbers = numbers[0:-1]
        if not price:
            # print("*price:",price)
            # print("len:",len(numbers[-1]['text']))
            if(len(numbers)):
                price = numbers[-1]['text']
                m = re.match(r'(\d+\.\d{2})\d*(\d{2})$', price)
                if m and not ratio:
                    price = m.group(1)
                    ratio = m.group(2) + '%'

                numbers = numbers[0:-1]

        numlen = len(numbers)
        if numlen == 3:
            specs = numbers[0]['text']
            amount = numbers[1]['text']
            uprice = numbers[2]['text']
        elif numlen == 2:
            num1 = numbers[0]['text']
            num2 = numbers[1]['text']

            if abs(float(num1) * float(num2) - float(price)) < 0.01:
                specs = ''
                amount = num1
                uprice = num2
            elif abs(float(num2) - float(price)) < 0.01:
                specs = num1
                amount = '1'
                uprice = num2
            else:
                specs, amount, uprice = get_amount_uprice(price, num2, num1)

        elif numlen == 1:
            specs = ''
            num = numbers[0]['text']
            if abs(float(num) - float(price)) < 0.01:
                amount = '1'
                uprice = num
            else:
                specs, amount, uprice = get_amount_uprice(price, num)

            if not amount:
                if uprice:
                    amount = str(int(float(price) / float(uprice) + 0.5))
                else:
                    amount = num

    if not ratio and price and tax:
        ratio = str(int(float(tax) * 100 / float(price) + 0.5)) + '%'

    return specs, amount, uprice, price, ratio

def get_date(line, price, ratio):
    sdate = ''
    edate = ''

    pattern = re.compile(r'\-?[\dBG:]+\.?[\dBG]*$')
    numbers = []

    for d in line:
        if pattern.match(d['text']):
            d['text'] = d['text'].replace('B','8').replace('G', '6').replace(':','')
            numbers.append(d)

    if len(numbers):
        for n in numbers:
            line.remove(n)

        # preprocess_number(numbers)
        numbers = sorted(numbers, key=lambda x: x['cx'] - x['w'] / 2)
        if not ratio and re.match(r'^\d{2,3}$', numbers[-1]['text']):
            ratio = numbers[-1]['text']
            ratio = ratio[:-1] + '%'
            numbers = numbers[0:-1]
        if not price:
            price = numbers[-1]['text']
            m = re.match(r'(\d+\.\d{2})\d*(\d{2})$', price)
            if m and not ratio:
                price = m.group(1)
                ratio = m.group(2) + '%'

            numbers = numbers[0:-1]

        numlen = len(numbers)
        if numlen == 2:
            sdate = numbers[0]['text']
            edate = numbers[1]['text']
        elif numlen == 1:
            edate = numbers[0]['text']

    return sdate, edate, price, ratio

def get_platenum_cartype(line, sdate, edate):
    platenum = ''
    cartype = ''

    pattern = re.compile(r'([\u4e00-\u9fa5]+)(\d{8,})$')
    if len(line) == 2:
        platenum = line[0]['text']
        cartype = line[1]['text']
    elif len(line) == 1:
        if not sdate:
            cartype = line[0]['text']
        else:
            platenum = line[0]['text']

    if cartype and not sdate:
        m = pattern.match(cartype)
        if m:
            cartype, sdate = m.group(1), m.group(2)
            if len(sdate) > 8 and not edate:
                edate = sdate[8:]
                sdate = sdate[:8]

    return platenum, cartype, sdate, edate


def preprocess_number(numbers):
    number = [i for i,n in enumerate(numbers) if n['text'].find(':')>-1]
    adds = []
    removes = []
    for i in number:
        d = numbers[i]
        text = d['text']
        splits = text.split(':')
        d1 = dict(d)
        d1['text'] = splits[0]
        d1['w'] = d['w'] * len(d1['text']) / len(text)
        d1['cx'] = d['cx'] - d['w'] / 2 + d1['w'] / 2

        d2 = dict(d)
        d2['text'] = splits[1]
        d2['w'] = d['w'] * len(d2['text']) / len(text)
        d2['cx'] = d['cx'] + d['w'] / 2 - d2['w'] / 2

        removes.append(d)
        adds.extend([d1, d2])

    for d in removes:
        numbers.remove(d)
    numbers.extend(adds)

def get_amount_uprice(price, upricecand, amtcand=None):
    price = float(price)
    specs = ''
    amount = ''
    uprice = ''

    copyprice = upricecand
    dotplace = upricecand.find('.')
    if dotplace > 0:
        upricecand = upricecand[:dotplace] + upricecand[dotplace + 1:]

    if amtcand:
        upr = str(math.trunc(float(price) / float(amtcand) * 100))
        idx = upricecand.find(upr)
        if idx >= 0:
            amount = amtcand
            upricecand = upricecand[idx:]
            dot = len(upr[:-2])
            uprice = upricecand[:dot] + '.' + upricecand[dot:]

    if not uprice:
        end = dotplace - 1 if dotplace else len(upricecand) - 2
        for idx in range(2, end):
            amt = int(upricecand[:idx])
            upr = upricecand[idx:]
            if not amt:
                break

            calcupr = price / amt
            if calcupr < 1:
                break
            dot = str(calcupr).find('.')
            if dot > len(upr):
                break
            upr = float(upr[0:dot] + '.' + upr[dot:])
            if abs(upr - calcupr) < 1:
                amount = str(amt)
                uprice = str(upr)
                break
            # upr = str(math.trunc(price / amt * 100))
            # if len(upr) < 3:
            #     break
            # i = upricecand.find(upr, idx)
            # if i > 0:
            #     amount = upricecand[0:i]
            #     uprice = upricecand[i:]
            #     dotplace = str(price / int(amount)).find('.')
            #     uprice = uprice[:dotplace] + '.' + uprice[dotplace:]
            #     break

        if not uprice:
            m = re.match(r'(\d+0+)([1-9]\d*\.\d+)', copyprice)
            if m:
                amount = m.group(1)
                uprice = m.group(2)
            else:
                uprice = copyprice
                if amtcand:
                    amount = amtcand
        else:
            if amtcand:
                specs = amtcand

    return specs,amount,uprice

def get_specs_unit(line, specs):
    unit = ''
    linelen = len(line)
    if linelen == 2:
        specs = line[0]['text']
        unit = line[1]['text']
    if linelen == 1:
        text = line[0]['text']
        if specs:
            unit = text
        else:
            if len(text) == 1:
                unit = text
            else:
                specs = text

    return specs,unit

def is_wrapped_title(data, line, boundary):
    res = False
    xmin = boundary[0]
    dx = abs(data['cx'] - data['w'] / 2 - xmin) # 该data的左边与最左边的差的绝对值
    text = data['text'] #
    if len(text) == 0: # 自己加的
        return res

    if dx < LEFT_MARGIN and text[0] != '*': # 偏左并且开头不是*
        res = True

    return res

def check_title(line, data, start, end, boundary):
    xmin = boundary[0]

    lf = min(line, key=lambda d:d['cx'] - d['w'] / 2)
    dx = abs(lf['cx'] - lf['w'] / 2 - xmin)
    if dx > LEFT_MARGIN:
        for d in data[start:end]:
            dx = abs(d['cx'] - d['w'] / 2 - xmin)
            if dx < LEFT_MARGIN:
                iou = [calc_axis_iou(d,l,1) for l in line]
                if np.mean(iou) > 0.3:
                    line.append(d)
                    data.remove(d)
                    break

def check_wrap_title(res, wraptitles, line=None):
    title = res[-1][0]
    wraplen = len(wraptitles)
    if wraplen:
        idx = 0
        if not line:
            wrap = wraptitles[:]
        else:
            wrap = []
            ref = min(line, key=lambda x:x['cx']-x['w']/2)
            for i,w in enumerate(wraptitles):
                if w['cy'] < ref['cy']:
                    wrap.append(w)
                else:
                    break

        if len(wrap):
            del wraptitles[0:len(wrap)]
            title = reduce(lambda a,b:a+b, [w['text'] for w in wrap], title)
            res[-1][0] = title


def get_basic_boundary(data):
    indexes = [i for i, d in enumerate(data) if re.search(r'开.?日期|票日期|校.?码|20.?.?年?.?.?月', d['text'])]
    # print(data[71])
    if len(indexes):
        end = max(indexes)
    else:
        end = 8
    lt = min(data[:end+1]+data[-10:], key=lambda x:x['cx']-x['w']/2)
    rt = max(data[:end+1]+data[-10:], key=lambda x:x['cx']+x['w']/2)

    left = lt['cx'] - lt['w'] / 2
    right = rt['cx'] + rt['w'] / 2

    return (0, end, left, right)

def get_basic_checkcode(basic):
    checkcode = ''
    candidates = [d for d in basic if re.search(r'^te校.?码*', d['text'])]
    if len(candidates):
        m = re.match(r'^校.?码.*?(\d+)', candidates[0]['text'])
        checkcode = m.group(1) if m else ''

    return checkcode
def get_basic_checkcode2(date):
    checkcode = ''
    candidates = [d for d in date if re.search(r'^校.?码*', d['text'])] # 找到，切只能找到一个
    if len(candidates) <=0:
        return ''
    # indexs = [idx for idx,d in enumerate(date) if re.search(r'^校.?码*', d['text'])]
    indexs = [i for i, d in enumerate(date) if re.search(r'^校.?码*', d['text'])]

    # 整体框的时候
    if candidates[0]['text'].find(':') >= 0:
        checkcode = candidates[0]['text'].split(':')[1]
    if len(checkcode)==20:
        return checkcode

    if len(candidates) == 1: # 不是整体框的时候
        r = indexs[0] + 5
        l = max(indexs[0]-5,0)

        can = date[l:r]
        x = float(candidates[0]['cx']) + float(candidates[0]['w'])/2
        y = float(candidates[0]['cy']) + float(candidates[0]['h'])/2
        distance_min = 10000
        index = 0
        for i in range(0,len(can)): # 先找到一个最小的离验证码框
            x_tem = float(can[i]['cx']) - float(can[i]['w'])/2
            y_tem = float(can[i]['cy']) + float(can[i]['h'])/2
            distance_tem = math.sqrt(math.pow((x - x_tem),2) + math.pow((y-y_tem),2))
            if distance_tem < distance_min:
                distance_min = distance_tem
                index = i
        flags = []
        flags.append(index)
        checkcode += can[index]['text']
        for i in (0,4):
            if(len(checkcode)>=20):
                break
            x = float(can[index]['cx']) + float(can[index]['w']) / 2
            y = float(can[index]['cy']) + float(can[index]['h']) / 2
            distance_min = 10000
            for j in range(0,len(can)):
                if j not in flags:
                    x_tem = float(can[j]['cx']) - float(can[j]['w']) / 2
                    y_tem = float(can[j]['cy']) + float(can[j]['h']) / 2
                    distance_tem = math.sqrt(math.pow((x - x_tem), 2) + math.pow((y - y_tem), 2))
                    if distance_tem < distance_min:
                        distance_min = distance_tem
                        index = j
            if can[index]['text'].isdigit():
                checkcode += can[index]['text']
                flags.append(index)

        #  最后一次检查 :
        if checkcode.find(':') >= 0:
            checkcode = checkcode.split(':')[-1]

        # m = re.match(r'^校.?码.*?(\d+)', candidates[0]['text'])
        # checkcode = m.group(1) if m else ''

    return checkcode
PROVINCE = ['河北','山西','辽宁','吉林','黑龙江','江苏','浙江','安徽','福建','江西','山东','河南','湖北','湖南','广东','海南','四川','贵州','云南','陕西']

def get_basic_type(basic):
    type = ''
    title = ''
    elec = '电子' if len([d for d in basic if re.search(r'发票代码', d['text'])])>0 else ''

    candidates = [d for d in basic if re.search(r'.*(专?用?|通)?发票', d['text'])]
    if len(candidates):
        text = candidates[0]['text']

        if text.find('用') >= 0 or text.find('专') >= 0:
            type = elec + '专用发票'
        else:
            type = elec + '普通发票'

        suffix = '增值税' + type
        title = suffix
        # ---新加的
        for p in PROVINCE:
            indexs = [d for d in basic if re.search(p, d['text'])]
            if len(indexs):
                title = p + suffix
                return type,title
        # ----
        if text[:2] in PROVINCE:
            title = text[:2] + suffix
        elif text[:3] in PROVINCE:
            title = text[:3] + suffix
        else:
            titles = [d for d in basic if re.search(r'^.*增值?', d['text'])]
            if len(titles):
                title = titles[0]['text']
                i = title.find('增')
                title = title[:i] + suffix
            else:
                i = basic.index(candidates[0])
                titles = [basic[i-1], basic[i+1]]
                for t in titles:
                    if re.match(r'[\u4e00-\u9fa5]{2,3}', t['text']): # duociyiju
                        # title = t['text'] + suffix
                        break
    else:
        for p in PROVINCE:
            indexs = [d for d in basic if re.search(p, d['text'])]
            if len(indexs):
                title = p + '增值税' + '发票'
                return type, title
    return type,title

def get_basic_title(basic, type):
    title = ''
    elec = '电子' if len([d for d in basic if re.search(r'发票代码', d['text'])]) > 0 else ''

    candidates = [d for d in basic if re.search(r'^.*增值?', d['text'])]
    if len(candidates):
        title = candidates[0]['text']
        i = title.find('增')
        title = title[:i] + '增值税' + elec + type

    return title

def get_basic_date(basic):
    date = ''
    # candidates = [d for d in basic if re.search(r'开?票?日期', d['text'])]
    candidates = [d for d in basic if re.search(r'20?.?.?年?.?.?月', d['text'])]
    if len(candidates):# 原来的Pse wangluo shi zhengti kuangde
        date = candidates[0]['text']
        date = re.sub(r'开?票?日期:?', '', date)

    return date

def get_basic_code(basic):
    code = ''
    candidates = [d for d in basic if re.search(r'(发票代码:?\d+)|(^\d{10,12}$)', d['text'])]
    if len(candidates):
        code = max(candidates, key=lambda x:x['cx']+x['w']/2)
        m = re.match(r'.*?(\d+)$', code['text'])
        code = m.group(1) if m else ''

    return code

def get_basic_sn(basic):
    sn = ''
    candidates = [d for d in basic if re.search(r'(发票号码:?\d+)|(^\d{8}$)', d['text'])]
    if len(candidates):
        code = max(candidates, key=lambda x: x['cx'] + x['w'] / 2)
        m = re.match('.*?(\d+)$', code['text'])
        sn = m.group(1) if m else ''

    return sn

def get_basic_payee(data):
    payee = ''
    candidates = [d for d in data if re.search(r'收款人', d['text'])]
    if len(candidates):
        payee = max(candidates, key=lambda x: x['cy'])
        payee = re.sub(r'收款人:?', '', payee['text'])

    return payee


def get_basic_reviewer(data):
    reviewer = ''
    candidates = [d for d in data if re.search(r'(复核|发校)', d['text'])]
    if len(candidates):
        reviewer = max(candidates, key=lambda x: x['cy'])
        reviewer = re.sub(r'(复核|发校):?', '', reviewer['text'])

    return reviewer


def get_basic_drawer(data):
    drawer = ''
    candidates = [d for d in data if re.search(r'开票人', d['text'])]
    if len(candidates):
        drawer = max(candidates, key=lambda x: x['cy'])
        drawer = re.sub(r'开票人:?', '', drawer['text'])

    return drawer

def parse_person(text):
    if text.find(':') >= 0:
        text = text.split(':')[1]
    else:
        text = re.sub(r'.*((开?票?人)|(复?核)|(复)|(收?款?人))', '', text)
    return text

def get_basic_person(data, boundary):
    payee = ''
    reviewer = ''
    drawer = ''

    payee_text = ''
    reviewer_text = ''
    drawer_text = ''

    xmin = boundary[0]

    rear = data[-12:] #范围大一点，防止盖章和备注信息的影响
    indexes = [i for i,d in enumerate(rear) if re.search(r'收?款|开票人|复?核|人:|复.:|校:', d['text'])]
    finded = len(indexes)
    if finded < 3:
        s = min(indexes) - 3
        e = min(10, max(indexes)+3)
        for i in range(s,e):
            if i not in indexes:
                text = rear[i]['text']
                l = len(text)
                if (text.find(':') > -1 and l < 8) or (text.find('钠售') < 0 and l < 4 and l > 1):
                    indexes.append(i)

        candidates = [rear[i] for i in indexes]
        if len(candidates):
            candidates = sorted(candidates, key=lambda x:x['cx']-x['w']/2)
            left = candidates[0]['cx'] - candidates[0]['w'] / 2 - xmin
            if left < 25:
                payee = candidates[0]
                candidates.pop(0)

            s = max([i for i, d in enumerate(data) if re.search(r'\(?大写\)?', d['text'])],default=0)
            e = min([i for i, d in enumerate(data) if re.search(r'\(?小写\)?.*(￥|Y|羊)?\d+', d['text'])],default=0)
            ref = data[e]
            s, e = (e-1, s+2) if e < s else (s, e+1)

            refs = sorted([data[i] for i in range(s,e) if calc_axis_iou(ref, data[i], 1)>0.01], key=lambda x: x['cx']-x['w']/2)
            if len(refs) >= 3:
                idx = refs.index(ref)
                ref = refs[idx-1]
                rl = ref['cx'] - ref['w'] / 2
                rr = ref['cx'] + ref['w'] / 2
                for c in candidates:
                    cl = c['cx'] - c['w'] / 2
                    cr = c['cx'] + c['w'] / 2
                    if cr > rr:
                        drawer = c
                        break
                    elif rl < c['cx'] and rr > c['cx']:
                        reviewer = c

    elif finded == 3:
        payee,reviewer,drawer = sorted([rear[i] for i in indexes],key=lambda x:x['cx']-x['w']/2)
        # payee_text = parse_person(payee['text'])
        payee_text = payee['text']
        if len(payee_text)>4:
            payee_text = payee_text[4:]
        else:
            x = float(payee['cx']) + float(payee['w'])/2
            y = float(payee['cy']) + float(payee['h'])/2 # 右下坐标

            index =-1
            distance_min = 100000
            for i in range(len(data)-10,len(data)):
                tem_payee = data[i]
                x_tem = float(tem_payee['cx']) - float(tem_payee['w'])/2
                y_tem = float(tem_payee['cy']) + float(tem_payee['h']) / 2  # 左下坐标
                distance_tmp = math.sqrt(math.pow((x - x_tem), 2) + math.pow((y - y_tem), 2))
                if distance_tmp < distance_min and len(tem_payee['text']) and distance_tmp <60:
                    index = i
                    distance_min = distance_tmp
            if index != -1:
                payee = data[index]
                payee_text = payee['text']

#--------------------------------------------
        # reviewer_text = parse_person(reviewer['text'])
        reviewer_text = reviewer['text']
        if len(reviewer_text) > 3:
            reviewer_text = reviewer_text[3:]
        else :
            x = float(reviewer['cx']) + float(reviewer['w'])/2
            y = float(reviewer['cy']) + float(reviewer['h'])/2

            index = -1
            distance_min = 100000
            for i in range(len(data)-10,len(data)):
                tem_reviewer = data[i]
                x_tem = float(tem_reviewer['cx']) - float(tem_reviewer['w'])/2
                y_tem = float(tem_reviewer['cy']) + float(tem_reviewer['w'])/2
                distance_tmp = math.sqrt(math.pow((x - x_tem),2) + math.pow((y - y_tem),2))
                if distance_tmp < distance_min and len(tem_reviewer['text']) and distance_tmp <60:
                    distance_min = distance_tmp
                    index  = i
            if index != -1:
                reviewer = data[index]
                reviewer_text = reviewer['text']

#--------------------------------------------------------
        # drawer_text = parse_person(drawer['text'])
        drawer_text = drawer['text']

        if len(drawer_text) > 4:
            drawer_text = drawer_text[4:]
        else:
            x = float(drawer['cx']) + float(drawer['w'])/2
            y = float(drawer['cy']) + float(drawer['h'])/2

            index = -1
            distance_min = 100000
            for i in range(len(data)-10,len(data)):
                tem_drawer = data[i]
                x_tem = float(tem_drawer['cx']) - float(tem_drawer['w'])/2
                y_tem = float(tem_drawer['cy']) + float(tem_drawer['h'])/2
                distance_tmp = math.sqrt(math.pow((x - x_tem),2) + math.pow((y - y_tem),2))
                if distance_min > distance_tmp and len(tem_drawer['text']) and distance_tmp <60:
                    distance_min = distance_tmp
                    index = i
            if index != -1:
                drawer = data[index]
                drawer_text = drawer['text']
    return payee_text,reviewer_text,drawer_text



    if payee:
        payee = parse_person(payee['text'])
    if reviewer:
        reviewer = parse_person(reviewer['text'])
    if drawer:
        drawer = parse_person(drawer['text'])

    return payee, reviewer, drawer

def getBasics(data):
    s, e, left, right = get_basic_boundary(data)
    basic = data[s:e+1]

    # checkcode = get_basic_checkcode(basic) #
    checkcode = get_basic_checkcode2(data) # 只有电子发票有验证码
    type, title = get_basic_type(basic)
    # title = basic_title(basic, checkcode, type)
    code = get_basic_code(basic)
    sn = get_basic_sn(basic)
    date = get_basic_date(basic)

    # payee = basic_payee(data[-10:])
    # reviewer = basic_reviewer(data[-10:])
    # drawer = basic_drawer(data[-10:])
    payee, reviewer, drawer = get_basic_person(data, [left, right])

    res = [[{'name': r'发票类型','value': type},
            {'name': r'发票名称','value': title},
            {'name': r'发票代码','value': code},
            {'name': r'发票号码','value': sn},
            {'name': r'开票日期','value': date},
            {'name': r'校验码','value': checkcode},
            {'name': r'收款人','value': payee},
            {'name': r'复核','value': reviewer},
            {'name': r'开票人','value': drawer}]]
    del data[:e-1]
    return res
    # return {"type":type, "title":title, "code":code, "sn":sn, "date":date, "checkcode":checkcode, "payee":payee, "reviewer":reviewer, "drawer":drawer}

def get_buyer_boundary(data):
    indexes = [i for i, d in enumerate(data) if isTitle(d['text'])]
    end = min(indexes) # 为什么取最小  因为去tittle之前的

    indexes = [i for i, d in enumerate(data) if i < end and re.search(r'(开票日期)|(校.?码)|(机器编号)|(20.?.?年.?.?月)', d['text'])]
    # print("len(indexes):",len(indexes))
    start = max(indexes,default=0) + 1 # 取 上面日期范围中最大的

    indexes = [i for i, d in enumerate(data[start:end]) if calc_axis_iou(data[start-1], d, 1) > 0.3]
    if len(indexes):
        start = start + max(indexes) + 1

    return start, end

def get_buyer_name(buyer):
    indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{6,}', d['text'])]
    # indexes = [i for i, d in enumerate(buyer) if re.search(r'^称:[\u4e00-\u9fa5]{6,}', d['text'])]
    if len(indexes):
        index = indexes[0]
    else:
        indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{6,}', d['text'])]
        index = indexes[0]
    name = buyer[index]
    text = name['text']
    if text.find(':') >= 0:
        name = text.split(':')[-1]
    else:
        name = re.sub(r'^[^\u4e00-\u9fa5]+?', '', text)
        name = re.sub(r'^称', '', name)

    return name, index

def get_buyer_taxnumber(buyer):
    indexes = [i for i, d in enumerate(buyer) if re.search(r'[0-9A-Z]{14,}', d['text'])]
    index = indexes[0]
    taxnumber = buyer[index]
    text = taxnumber['text']
    if text.find(':') >= 0:
        taxnumber = text.split(':')[1]
    else:
        taxnumber = re.sub(r'^[^0-9A-Z]+?', '', text)

    return taxnumber, index

def get_buyer_address(buyer):
    address = ''
    index = 0

    indexes = [i for i, d in enumerate(buyer) if re.search(r'电话:[\u4e00-\u9fa5]{7,}', d['text'])]
    if not len(indexes):
        indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{7,}', d['text'])]

    if len(indexes):
        index = indexes[0]
        address = buyer[index]
        text = address['text']
        if text.find(':') >= 0:
            address_tem = text.split(':')
            address = ''
            for i in range(1,len(address_tem)):
                # print(address_tem[i])
                address += str(address_tem[i])
        else:
            address = text

        if not re.search(r'[0-9\-]{11,}$', address):
            # indexes = [i for i, d in enumerate(buyer) if re.match(r'\d+$', d['text']) and i > index]
            indexes = [i for i, d in enumerate(buyer) if re.match(r'[0-9\-]{6,}$', d['text'])]
            x = float(buyer[index]['cx']) + float(buyer[index]['w']/2)
            y = float(buyer[index]['cy']) + float(buyer[index]['h']/2)
            distance_min = 100000
            for i in indexes:
                x_tem = float(buyer[i]['cx']) - float(buyer[i]['w'] / 2)
                y_tem = float(buyer[i]['cy']) + float(buyer[i]['h'] / 2)
                distance_tem = math.sqrt(math.pow((x-x_tem),2) + math.pow((y-y_tem),2))
                if distance_min > distance_tem:
                    index = i
                    distance_min = distance_tem
            address += buyer[index]['text']

             # address += buyer[index]['text']

            # if len(indexes):
            #     index = indexes[0]
            #     address += buyer[index]['text']

        for prov in PROVINCE:
            idx = address.find(prov)
            if idx > 0:
                address = address[idx:]
                break

    return address, index

def get_buyer_account(buyer):
    account = ''

    indexes = [i for i, d in enumerate(buyer) if re.search(r'账号:[\u4e00-\u9fa5]{7,}', d['text'])]
    if not len(indexes):
        indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{7,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        account = buyer[index]
        text = account['text']
        if text.find(':') >= 0:
            account = text.split(':')[1]
        else:
            account = text

        if not re.search(r'\d{12,}$', account):
            # indexes = [i for i, d in enumerate(buyer) if re.match(r'\d{12,}$', d['text']) and i > index]
            indexes = [i for i, d in enumerate(buyer) if re.match(r'\d{12,}$', d['text'])]
            if len(indexes):
                index = indexes[0]
                account += buyer[index]['text']

        idx = account.find(r'账号')
        if idx >= 0:
            account = account[idx+2:]

    return account

def getBuyer(data):
    start, end = get_buyer_boundary(data)
    buyer = data[start:end]

    name, index = get_buyer_name(buyer)
    buyer = buyer[index+1:]

    taxnum, index = get_buyer_taxnumber(buyer)
    buyer = buyer[index+1:]

    address, index = get_buyer_address(buyer)
    buyer = buyer[index+1:]

    account = get_buyer_account(buyer)

    res = [[{'name':r'名称', 'value':name},
            {'name':r'纳税人识别号', 'value':taxnum},
            {'name':r'地址、电话', 'value':address},
            {'name':r'开户行及账号','value':account}]]
    return res
    # return {"name":name, "taxnum":taxnum, "address":address, "account":account}

def isVehicle(data):
    ret = False
    l = len(data)
    for d in data[:int(l/2)]:
        if re.search(r'项目|类型|车牌|通行日期', d['text']):
            ret = True
            break

    return ret

def is_wrapped_title2(data, line, content, next, boundary) :
    """

    :param data: ct
    :param line: one line have
    :param content:
    :param next: save as now data next index
    :param boundary: left and right
    :return:
    """
    ret = False
    xmin = boundary[0]
    dx = abs(data['cx'] - data['w'] / 2 - xmin)
    text = data['text']

    if dx < LEFT_MARGIN :
        if len(text) < 6 and len(line) : # len(tittle)==6 and
            ret = True
        else :
            if not re.search(r'\*[\u4e00-\u9fa5]', text) : # not *zhongwen
                if next == len(content) : # last one
                    if (not len(line) or calc_axis_iou(data, line) > 0.2) :
                        ret = True
                else : # is not last one
                    end = min([next + 8, len(content)])
                    neighbor = None
                    title = None
                    for c in content[next :end] :
                        if calc_axis_iou(c, data) > 0.2 :
                            title = c
                        else :
                            if neighbor is None:
                                if c['text'] != '':
                                    neighbor = c
                            else :
                                dx = c['cx'] - data['cx']
                                x = neighbor['cx'] - data['cx']
                                y = calc_axis_iou(c, neighbor,1)
                                z = calc_axis_iou(c, neighbor)
                                #                                               must correct img_angle
                                if (neighbor['cx'] - data['cx'] > dx > 0) and calc_axis_iou(c, neighbor,1) > 0.3 and calc_axis_iou(c, neighbor) < 0.1 :
                                #   now neighbor is more near than before       in Y must be iou                    is not iou in X
                                    neighbor = c

                    if neighbor is None : # have not neighbor
                        ret = True
                    else :
                        y_iou = calc_axis_iou(neighbor, data, 1)
                        if y_iou > 0.3 : # neighbor and data have big iou
                            if len(line) and len(data) and len(neighbor):
                                if calc_axis_iou(neighbor, line, 1) > 0.1 and calc_axis_iou(neighbor,line) < 0.1 and calc_axis_iou(data, line) > 0.1 :
                                #  neighbor and before line have iou in Y        neighbor and before line have not iou in X     data and before line have iou in X
                                    ret = True
                        elif y_iou < 0.1 : # have little iou
                            ret = True
                        else : # have not iou in Y
                            if title is not None :
                                if calc_axis_iou(neighbor, title, 1) > 0.25 :
                                    ret = True
                            else :
                                if calc_axis_iou(data, line) > 0.2 :
                                    ret = True

    # if dx < LEFT_MARGIN and len(line) and calc_axis_iou(data, line) > 0.2 and (len(text) < 5 or not re.search(r'\*[\u4e00-\u9fa5]', text)):
    #     ret = True

    return ret


def getContent(data):
    res = []
    start,end,left,right = get_content_boundary(data)
    content = data[start+1:end]

    while len(content):
        left_tem = min(content, key = lambda x : x['cx'] - x['w'] / 2)
        if len(left_tem['text']) < 3 :
            content.remove(left_tem)
        else :
            break
    while len(content):
        right_tem = max(content, key = lambda x : x['cx'] + x['w'] / 2)
        if len(right_tem['text']) < 3 :
            content.remove(right_tem)
        else :
            break

    isvehicle = isVehicle(data) # 是否是私家车
    # top = min(content, key=lambda x:float(x['cy'])-float(x['h']/2))
    # bottom = max(content, key=lambda x: float(x['cy'])+float(x['h']/2))
    # lh = (bottom['cy'] + bottom['h']/2 - top['cy'] + top['h']/2) / 8

    lt = min(content, key=lambda x:float(x['cx'])-float(x['w']/2)) # 得到最左边的
    rb = max(content, key=lambda x: float(x['cx'])+float(x['w']/2)) # 得到最右边的

    # 下面这两行是根据中间的内容找的，不加就是根据 title 找的
    left = float(lt['cx'])-float(lt['w']/2) # 得到最左边的边
    right = float(rb['cx'])+float(rb['w']/2) # 得到最右边的边

    line = []
    wraptitle = [] #
    for idx,ct in enumerate(content):

        deal = False
        # iswrap = is_wrapped_title(ct, line, [left,right]) # ?? 可能是判断该行是不是之前的换行 True 是 上一行
        iswrap = is_wrapped_title2(ct, line, content, idx + 1, [left, right])
        if not iswrap and ct not in wraptitle:
            linelen = len(line)
            if linelen:
                y_ious = []
                for l in line:
                    x_iou = calc_axis_iou(l, ct) # l 是 dict,ct 是 dict
                    y_iou = calc_axis_iou(l, ct, 1)
                    y_ious.append(y_iou)
                    if x_iou > 0.3: # 交叉部分大就跳出
                        deal = True
                        break
                if not deal and np.mean(y_ious) < 0.05:# y的差距小 说明在一行
                    deal = True

            if deal == False:
                line.append(ct)
            else: # deal 为 true 已经跳出，说明当前的ct与之前的line里面的存在交叉
                #           存前idx  总
                check_title(line, content, idx+1, idx+4, [left,right])
                if len(res): # 如果结果中已经有值，可以把判断为换行的tittle加到上一行
                    check_wrap_title(res, wraptitle, line)
                parsed = parseLine(line, [left, right], isvehicle)
                res.append(parsed)

                line = [ct]
        else: # 是换行
            #--- shihuanhang qie huanhang de neirong bei fencheng duoge bufen
            if ct not in wraptitle:
                wraptitle.append(ct)
                start_idx = idx+1
                end_idx = idx + 3
                if end_idx > len(content):
                    end_idx = len(content)
                for i in range(start_idx,end_idx):
                    y_iou = calc_axis_iou(content[i], ct, 1)
                    if i != idx and calc_axis_iou(content[i], ct, 1) > 0.9:
                        wraptitle.append(content[i])
            #---
            # flag = 0

    if len(line) + len(wraptitle) >= 3:
        if len(res):
            check_wrap_title(res, wraptitle, line)
        if len(wraptitle):
            line.extend(wraptitle)
        parsed = parseLine(line, [left, right], isvehicle)
        res.append(parsed)

    ret = []
    calcprice = 0
    calctax = 0
    for r in res:
        if isvehicle:
            title, platenum, type, sdate, edate, price, ratio, tax = r
            ret.append([{'name': r'项目名称', 'value': title},
                        {'name': r'车牌号', 'value': platenum},
                        {'name': r'类型', 'value': type},
                        {'name': r'通行日期起', 'value': sdate},
                        {'name': r'通行日期止', 'value': edate},
                        {'name': r'金额', 'value': price},
                        {'name': r'税率', 'value': ratio},
                        {'name': r'税额', 'value': tax}])
        else:
            title, specs, unit, amount, uprice, price, ratio, tax = r
            ret.append([{'name': r'名称', 'value': title},
                        {'name': r'规格型号', 'value': specs},
                        {'name': r'单位', 'value': unit},
                        {'name': r'数量', 'value': amount},
                        {'name': r'单价', 'value': uprice},
                        {'name': r'金额', 'value': price},
                        {'name': r'税率', 'value': ratio},
                        {'name': r'税额', 'value': tax}])
        # print("price:",price)
        if len(price):
            calcprice += float(price)
        calctax += float(tax) if is_number(tax) else 0

    calctotal = '%.2f' % (calcprice + calctax)
    calcprice = '%.2f' % calcprice
    if calctax <= 0.001:
        calctax = '***'
    else:
        calctax = '%.2f' % calctax
    return ret, (calctotal, calcprice, calctax)

def get_seller_boundary(data):
    s = max([i for i, d in enumerate(data) if re.search(r'\(?大写\)?|价税合?计?', d['text'])])
    # e = min([i for i, d in enumerate(data[s-4:]) if re.search(r'\(?小写?\)?', d['text'])])
    e = min([i for i, d in enumerate(data) if re.search(r'(￥|Y|羊)[0-9]+(.[0-9]{2})?$|\(?小写\)?', d['text'])])
    # e = min([i for i, d in enumerate(data) if re.search(r'\(?小写\)?.*(￥|Y|羊)?\d+', d['text'])])


    start = max([s,e]) + 1
    # start = max([i for i, d in enumerate(data) if re.search(r'\(?小写\)?|\(?大写\)?|价税合?计?|￥[0-9]+(.[0-9]{2})?$|Y[0-9]+(.[0-9]{2})?$|羊[0-9]+(.[0-9]{2})?$', d['text'])]) + 1
    # if abs(s-e) == 1:
    #     start = start + 1 # 为什么 +1
    end = len(data) - 2

    benchmark = ['仟', '佰', '拾', '亿', '仟', '佰', '拾', '万', '仟', '佰', '拾', '圆', '角', '分','零','壹','贰','叁','肆','伍','陆','柒','捌','玖','拾','整']

    text = data[start]['text']
    i = 0
    for t in text:
        if t in benchmark:
            i += 1

    if i>2:
        return start + 1,end
    else:
        return start, end

def get_seller_name(buyer):
    name = ''
    index = -1
#                                                             匹配中文
    indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{6,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        name = buyer[index]
        text = name['text']
        # -----
        if len(text):
            benchmark = ['仟', '佰', '拾', '亿', '仟', '佰', '拾', '万', '仟', '佰', '拾', '圆', '角', '分','零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾','整']
            flag = 0
            for i in range(0,len(text)):
                if text[i] in benchmark:
                    flag += 1
            if flag > 3 and len(indexes) > 1:
                index = indexes[1]
                name = buyer[index]
                text = name['text']
        # ---
        if text.find(':') >= 0:
            name = text.split(':')[1]
        else:
            name = re.sub(r'^[^\u4e00-\u9fa5]+?', '', text)
            name = re.sub(r'^称|你', '', name)

    return name, index

def get_seller_taxnumber(buyer,RIGHT_MOST):
    taxnumber = ''
    index = -1

    # 排除备注中的信息，以图片的中间为限
    indexes = [i for i, d in enumerate(buyer) if float(d['cx']) - float(d['w'])/2 <= RIGHT_MOST/2 and re.search(r':[0-9A-Z]{16,}|^[0-9A-Z]{16,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        taxnumber = buyer[index]
        text = taxnumber['text']
        if text.find(':') >= 0:
            taxnumber_tem = text.split(':')
            taxnumber = ''
            for i in range(0,len(taxnumber_tem)):
                if re.search(r'^[0-9A-Z]{1,}',taxnumber_tem[i]):
                    taxnumber += taxnumber_tem[i]
        else:
            taxnumber = re.sub(r'^[^0-9A-Z]+?', '', text)

    return taxnumber, index

def get_seller_address(buyer,RIGHT_MOST):
    address = ''
    index = -1

#                                                              匹配中文
    indexes = [i for i, d in enumerate(buyer) if float(d['cx']) - float(d['w'])/2<= RIGHT_MOST/2 and re.search(r'[\u4e00-\u9fa5]{6,}', d['text']) and not re.search('识别号',d['text'])]
    index_add = sorted([d for i, d in enumerate(buyer) if re.search('地址|电话',d['text'])],key=lambda x: x['cx'] + x['w'] / 2,reverse=True)
    if len(index_add):
        index = buyer.index(index_add[0])
        address = index_add[0]
        text = address['text']
        if text.find(':') >= 0:
            address = text.split(':')[-1]
        else:
            address = ''

        address = re.sub(r'^地址、?电话', '', address)
        index_max = int(index)
        bre = 0
        mark = 0
        while True:  # forbid seller address three part
            if not re.search(r'[0-9\-]{11,}$', address):
                #---- deal with 8 -> B
                rear_text = address.replace('B','8')
                if re.search(r'[0-9\-]{11,}$', rear_text):
                    for i in range(len(address) - 1,-1,-1):
                        if re.search(r'[\u4e00-\u9fa5]',address[i]):
                            break
                        if address[i] == 'B':
                            address = address[:i] + '8' + address[i+1:]
                    break
                #----
                indexs = [i for i, d in enumerate(buyer)]
                if len(address) < 5 and mark == 0:
                    index_tem = get_min_distance_index(buyer[index], indexes, buyer)
                    mark = 1
                else:
                    indexes = indexs
                    index_tem = get_min_distance_index(buyer[index], indexes, buyer)
                indexes = indexs
                address += buyer[index_tem]['text']
                if index_tem > index_max:
                    index_max = index_tem
                index = index_tem
            else:
                break
            bre += 1
            if bre > 3:
                break
            # buyer.pop(index_tem)
        index = index_max

        for prov in PROVINCE:
            idx = address.find(prov)
            if idx > 0:
                address = address[idx:]
                break
        return address, index

    else:
        if len(indexes):
            index = indexes[0]
            address = buyer[index]
            text = address['text']
            if text.find(':') >= 0:
                address = text.split(':')[1]
            else:
                address = text
            address = re.sub(r'^地址、?电话', '', address)

            index_max = int(index)
            bre = 0
            while True: # forbid seller address three part
                if not re.search(r'[0-9\-]{11,}$',address):
                    indexs = [i for i,d in enumerate(buyer)]
                    index_tem = get_min_distance_index(buyer[index],indexs,buyer)
                    address += buyer[index_tem]['text']
                    if index_tem >index_max:
                        index_max = index_tem
                    index = index_tem
                else:
                    break
                bre += 1
                if bre > 3:
                    break
            index = index_max

            for prov in PROVINCE:
                idx = address.find(prov)
                if idx > 0:
                    address = address[idx:]
                    break

    return address, index

def get_seller_account(buyer,RIGHT_MOST):
    account = ''
    indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{7,}', d['text']) and d['cx'] < RIGHT_MOST / 2 ]
    # indexes = [i for i, d in enumerate(buyer) if re.search(r'行及账',d['text'])]
    if len(indexes):
        index = indexes[0]
        account = buyer[index]
        text = account['text']
        if text.find(':') >= 0:
            splittxt = text.split(':')
            account = ''.join(splittxt[1:])
        else:
            account = text

        if not re.search(r'\d{12,}$', account):
            # indexes = [i for i, d in enumerate(buyer) if re.match(r'[0-9]{13,}$', d['text'])]
            indexes = [i for i, d in enumerate(buyer) if re.search(r'[0-9]{13,}$', d['text'])]
            ii = get_min_distance_index(buyer[index],indexes,buyer)
            # distance_min = 100000
            # ii = 0
            # for i in indexes:
            #     x_tem = float(buyer[i]['cx']) - float(buyer[i]['w']/2)
            #     y_tem = float(buyer[i]['cy']) + float(buyer[i]['h']/2)
            #     distance_tem = math.sqrt(math.pow((x-x_tem),2) + math.pow((y-y_tem),2))
            #     if distance_tem < distance_min:
            #         ii = i
            #         distance_min = distance_tem
            account += buyer[ii]['text']

            # if len(indexes):
            #     index = indexes[0]
            #     account += buyer[index]['text']

        idx = account.find(r'账号')
        if idx >= 0:
            account = account[idx+2:]

    return account

def getSeller(data):
    RIGHT_MOST = get_right_marge(data) # 找最右边的边
    start, end = get_seller_boundary(data) # 从大写/小写开始 到 倒数第二个结束
    seller = data[start:end]

    name, index = get_seller_name(seller)
    seller = seller[index + 1:]

    taxnum, index = get_seller_taxnumber(seller,RIGHT_MOST)
    seller = seller[index + 1:]
    address, index = get_seller_address(seller,RIGHT_MOST)
    seller = seller[index + 1:]

    account = get_seller_account(seller,RIGHT_MOST)

    res = [[{'name': r'名称', 'value': name},
            {'name': r'纳税人识别号', 'value': taxnum},
            {'name': r'地址、电话', 'value': address},
            {'name': r'开户行及账号', 'value': account}]]
    return res
    # return {"name": name, "taxnum": taxnum, "address": address, "account": account}

def get_summation_boundary(data):
    summation = [i for i, d in enumerate(data) if re.search(r'\(?大写\)?', d['text'])]
    summation.extend([i for i, d in enumerate(data) if re.search(r'\(?小写\)?.*(￥|Y|羊)?[\d\.]+$', d['text'])])
    summation.extend([i for i, d in enumerate(data) if re.search(r'(￥|Y|羊)[\d+\.]+$', d['text'])])

    start = min(summation) - 1
    end = max(summation) + 1

    return start, end

def check_price(price, calc):
    if price:
        price = re.sub(r'^\D+', '', price['text'])
        if re.search(r'[^\d\.]', price):
            price = calc
        else:
            idx = price.rfind(r'.')
            if idx <= 0:
                if len(price) > 2:
                    price = price[:-2] + '.' + price[-2:]
            else:
                price = price[:idx].replace(r'.', '') + (price[idx:] if len(price[idx:]) <= 3 else price[idx:idx + 3])

            if len(price) <= 2:
                price = calc
    else:
        price = calc

    return price

def getSummation(data, calcsum):
    benchmark = ['仟', '佰', '拾', '亿', '仟', '佰', '拾', '万', '仟', '佰', '拾', '圆', '角', '分']
    chinesedigit = ['零','壹','贰','叁','肆','伍','陆','柒','捌','玖','拾']
    tax = ''
    price = ''
    total = ''
    capital = ''

    calctotal, calcprice, calctax = calcsum
    _,_,_,right = get_content_boundary(data)

    start, end = get_summation_boundary(data)
    summation = data[start:end]

    prices = [d for d in summation if re.search(r'(￥|Y|羊)?[\d\.]+$', d['text'])]
    #--- 处理盖章乱盖 and summation
    if len(prices) > 3:
        prices = sorted(prices, key=lambda x: x['cx'],reverse=True)
        for i in range(len(prices) - 1, -1, -1):
            if not re.search(r'(￥|Y|羊)[\d\.]+$', prices[i]['text']):
                prices.pop(i)
            else: # 如果匹配到就break
                break
            if len(prices)==3:
                break
        if len(prices) > 3:
            Y = [d for i,d in enumerate(prices) if re.search(r'(￥|Y|羊)[\d\.]+$', d['text'])]
            if len(Y): # 处理税额的影响
                Y = sorted(Y,key=lambda x: x['cx'] + x['w'] / 2,reverse=True)
                tax_right = Y[0]

                if math.fabs(float(tax_right['cx']) + float(tax_right['w'])/2 - right) < RIGHT_MARGIN:
                    for j in range(len(prices)-1, -1, -1):
                        if tax_right != prices[j] and float(tax_right['cy'])>float(prices[j]['cy']):
                            cha_rate = calc_axis_iou(tax_right, prices[j], axis=0)
                            if calc_axis_iou(tax_right, prices[j], axis=0) > 0.5:
                                prices.pop(j)
                            if len(prices) == 3 :
                                break
            if len(Y): # 处理金额的影响
                Y = sorted(Y, key=lambda x: x['cx'] - x['w'] / 2)
                price_left = Y[0]
                if len(Y) > 1 and math.fabs(float(Y[-1]['cx']) + float(Y[-1]['w'])/2 - right) < RIGHT_MARGIN:
                    for j in range(len(prices) - 1 ,-1,-1):
                        if price_left != prices[j] and float(price_left['cy']) > float(prices[j]['cy']):
                            if calc_axis_iou(price_left, prices[j]) > 0.5:
                                prices.pop(j)
                            if len(prices) == 3:
                                break
                # 找一个条件说明是金额

    #---

    if len(prices): # 是不是该满足大于1
        prices = sorted(prices, key=lambda x: x['cy'], reverse=True) # big -> small
        p = prices[0]
        ll = calc_axis_iou(p, prices[1:], 1)
        if calc_axis_iou(p, prices[1:], 1) < 0.11:#  p 和 另外的数在y方向上都不存在交叉，说明 p 是 tatal
            total = p
            prices.remove(p)

    if len(prices):
        prices = sorted(prices, key=lambda x: x['cx'], reverse=True)
        p = prices[0]
        if abs(p['cx']+p['w']/2 - right) < RIGHT_MARGIN: # 最靠右的是tax
            tax = p
            prices.remove(p)

    if len(prices):
        price = prices[0]

    total = check_price(total, calctotal)
    price = check_price(price, calcprice)
    tax = check_price(tax, calctax)
    try:

        if total == calctotal and tax == calctax and math.fabs(float(calctax)+float(calcprice)-float(calctotal)) < 0.1: # 补丁 防止印章影响总的price
            price = calcprice
    except ValueError:
        print("float 转换类型出错")
    strtotal = re.sub(r'\.', '', total)
    bm = benchmark[-len(strtotal):]

    for (c, b) in zip(strtotal, bm):
        capital += chinesedigit[int(c)] + b

    if int(total[-2:]) == 0:
        capital = capital[:-4] + '整'
    capital = re.sub(r'零[仟佰拾角分]', '零', capital)
    capital = re.sub(r'零{2,}', '零', capital)
    capital = re.sub(r'零$', '', capital)
    capital = re.sub(r'零圆', '圆', capital)

    if capital[-1] != '整' and capital[-1] != '分':
        capital += '整'
    res = [[{'name': r'金额合计', 'value': price},
            {'name': r'税额合计', 'value': tax},
            {'name': r'价税合计(大写)', 'value': capital},
            {'name': r'价税合计(小写)', 'value': total}]]
    return res

def calc_axis_iou(a,b,axis=0):
    if isinstance(b, list): # b 是list类型时
        if axis == 0: # x 方向的交叉率
            #                     左                        右                     左                 右
            ious = [calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2]) for x in b] #
        else: # y fangxinag de jiaocha lv
            ious = [calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2]) for x in b]
        iou = max(ious)
    elif isinstance(a, list):
        if axis == 0:
            ious = [calc_iou([x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2]) for x in a]
        else:
            ious = [calc_iou([x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2]) for x in a]
        iou = max(ious)
    else: # a b 都不是 list类型
        if axis == 0:#
            iou = calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2])
        else:
            iou = calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2])
    return iou

def sort_result(data):
    data = sorted(data, key=lambda d: d['cy'])
    lines = []
    line = []
    for i in range(len(data)):
        d = data[i]
        if not len(line):
            line.append(d)
        else:
            iou_x = calc_axis_iou(d, line, 0)
            iou_y = calc_axis_iou(d, line, 1)
            if iou_y > 0.6 and iou_x < 0.1:
                line.append(d)
            else:
                line = sorted(line, key=lambda l:l['cx']-l['w']/2)
                lines.append(line)
                line = [d]

    if len(line):
        line = sorted(line, key=lambda l: l['cx'] - l['w'] / 2)
        lines.append(line)

    return lines


def formatResult(data):

    basic = getBasics(data)

    buyer = getBuyer(data)

    content,calcsum = getContent(data)
    seller = getSeller(data)
    summation = getSummation(data, calcsum)
    res = [{'title':r'发票基本信息', 'items':basic},
           {'title':r'购买方', 'items':buyer},
           {'title':r'销售方', 'items':seller},
           {'title':r'货物或应税劳务、服务', 'items':content},
           {'title':r'合计', 'items':summation}]
    return res
    # return {"basic":basic, "buyer":buyer, "content":content, "seller":seller}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=960,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.5,
                        help='min score')

    args = parser.parse_args()
    test(args)
