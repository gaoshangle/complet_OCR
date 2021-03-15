# -*- coding:utf-8 -*-
import math
import util
from apphelper.image import  calc_iou
from crnn import crnnRec
# from angle import *
# from eval.eval_business_license import evaluate

def recognize(im=None, path=None):
    ret = None
    try:
        file = None
        if im:
            pass
        elif path:
            # 提取文件路径
            # dir,base = os.path.split(path)
            # file,suffix = os.path.splitext(base)
            # dir = os.path.dirname(__file__)
            # tmpfile = os.path.join(dir, 'tmp/'+file+'-large'+suffix)
            # 修改图片大小和分辨率
            im = Image.open(path)
            file = os.path.basename(path)

        if im:
            dir = '/home/share/gaoluoluo/complete_ocr/data/images/tmp/'
            file = file if file is not None else 'tmp.jpg'
            tmpfile = os.path.join(dir, file)
            im.save(tmpfile)

            data = test(tmpfile)
            if data:
                ret = data
    except Exception as e:
        print(e)

    return ret


def get_min_distance_index(tem,indexes,near,maxRight = 100):
    x = float(tem['cx']) + float(tem['w']) / 2
    y = float(tem['cy']) + float(tem['h']) / 2
    distance_min = 100000
    ii = -1
    for i in range(0,len(near)):
        x_tem = float(near[i]['cx']) - float(near[i]['w']/2)
        y_tem = float(near[i]['cy']) + float(near[i]['h']/2)
        distance_tem = math.sqrt(math.pow((x-x_tem),2) + math.pow((y-y_tem),2))
        if distance_tem < distance_min and near[i]['cx'] < maxRight and len(near[i]['text']) and calc_axis_iou(near[i],tem,1) > 0.3 and float(tem['cx']) < float(near[i]['cx']):
            ii = i
            distance_min = distance_tem
        # if distance_tem < 30 and re.search(r'[\u4e00-\u9fa5]{6,}',near[i]['text']):
        #     return i
    return ii

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)

import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
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
        # print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        # print("checkpoint:",checkpoint)

        config = checkpoint['config']
        # print(checkpoint['config'])
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
        # assert os.path.exists(img_path), 'file is not exists'
        # img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
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
    parser.add_argument('--model_path', default=r'/home/share/gaoluoluo/模型/model50_ch_epoch510_latest.pth', type=str)
    # parser.add_argument('--input_folder', default='/home/share/gaoluoluo/invoices_gao_true/img/', type=str, help='img path for predict')
    parser.add_argument('--input_folder', default='/home/share/gaoluoluo/complete_businessLicense/input/img/', type=str, help='img path for predict')
    parser.add_argument('--img_correct',default='/home/share/gaoluoluo/dbnet/test/test_corre_input/',type=str,help='img_correct path for predict')
    # parser.add_argument('--input_folder',default='/home/share/gaoluoluo/dbnet/test/test_corre_input', type=str, help='img path for predict')
    # parser.add_argument('--input_folder', default='/home/share/gaoluoluo/complete_ocr/data/images/tmp', type=str,help='img path for predict')
    parser.add_argument('--output_folder', default='/home/share/gaoluoluo/dbnet/test/test_output/', type=str, help='img path for output')
    # parser.add_argument('--output_folder', default='/home/share/gaoluoluo/invoices_gao_true/outputs/', type=str, help='img path for output')
    parser.add_argument('--gt_txt', default='/home/share/gaoluoluo/complete_businessLicense/input/gt/', type=str, help='img 对应的 txt')
    parser.add_argument('--thre', default=0.1, type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_result', action='store_true', help='save box and score to txt file')
    parser.add_argument('--evaluate', nargs='?', type=bool, default=False,
                        help='evalution')
    args = parser.parse_args()
    return args

def test(tmpfile):

    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from dbnet.utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    # 初始化网络                                         0.1
    model = Pytorch_model(args.model_path, post_p_thre=args.thre, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)  # dbnet/test/input/
    # i = 0

    total_frame = 0.0
    total_time = []

    precisions = []

    names = []

    idCode = []
    names1 = []
    sex = []
    nation = []
    address = []
    brithday = []
    dataOfFirstIssue = []
    classType = []
    validPerios = []

    for ind in range(0,1):  # img_path /home/share/gaoluoluo/dbnet/test/input/2018实验仪器发票.jpg
        # print("img_path:",img_path)  /home/share/gaoluoluo/dbnet/test/input/2018实验仪器发票.jpg
        img_path = tmpfile
        print("\nimg_path:", img_path)
        # img_path = '/home/share/gaoluoluo/IDCard/corre_id/3.jpg'
        start_time = time.time()
        img = angle_corre(img_path)# 调整图片角度 wei调整
        # img = cv2.imread(img_path)
        img_path = args.img_correct + img_path.split('/')[-1] # After correct angle img path
        names.append(img_path.split('/')[-1])
        # names_tem.append(str(iii)+img_path.split('.')[-1])
        # iii += 1
        # print("img_path:",img_path)
        preds, boxes_list, score_list, t = model.predict(img, is_output_polygon=args.polygon)
        img_path1 = img_path
        boxes_list_save = boxes_list
        box = []
        for i in range(0,len(boxes_list)):
            for  j in range(0,len(boxes_list[0])):
                b = []
                b.append(np.float32(boxes_list[i][j][0]))
                b.append(np.float32(boxes_list[i][j][1]))
                box.append(b)
        boxes_list = boxes_list.reshape(-1,2)
        boxes_list = box
        #---hekuang  zhineng chuli yige kuang bei fencheng liangge bufen
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
        for i in range(0,len(points_kuang)):# 逆时针  1 -> 2 -> 3 -> 4
            point3 = points_kuang[i][2]
            start_point = i - 3
            end_point = i + 3
            if start_point < 0:
                start_point = 0
            if end_point > len(points_kuang):
                end_point = len(points_kuang)
            if i not in remove_mark:
                for j in range(start_point,end_point):
                    point4 = points_kuang[j][3]
                    min_dis = math.sqrt(math.pow((point3[0] - point4[0]),2) + math.pow((point3[1] - point4[1]),2))
                    Y_cha = math.fabs(point3[1] - point4[1])
                    if min_dis < 5 and Y_cha < 5 and j not in remove_mark and i != j : # 10 reasonable
                    # if min_dis < 0:
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
                        # print("i=",i," j=",j)
                        # print('x:',point4[0])
                        break
        remove_mark = sorted(remove_mark,reverse=True)
        # print("---------------------------")
        for _,i in enumerate(remove_mark):
            # print('x:',points_kuang[i][3][0])
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
        # ori_img = cv2.imread('/home/share/gaoluoluo/IDCard/outputs/part_id/0.png')
        # print(img_path1)
        rects = reversed(rects)
        result = crnnRec(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB), rects)
        # result = crnnRec_tmp(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB), rects)
        # result = list(reversed(result))

        # print("result:",result)
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
        # print("--------------")

        # print("-------------------11")
        result = formatResult(result)

    return result;
import re

def calc_axis_iou(a,b,axis=0):
    if isinstance(b, list): # b 是list类型时
        if axis == 0: # x 方向的交叉率
            #                     左                        右                     左                 右
            ious = [calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2]) for x in b] #
        else: # y 方向的交叉类
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

# ------------------------------------------------------------------------------------


def getValidPeriod(data):
    start_time = ''
    end_time = ''
    candidates = [d for d in data if re.search(r'\d{4}-|^(19|20|28)|-..-|.?..-..', d['text']) and len(d['text']) <= 13]
    if len(candidates):
        candidates = sorted(candidates, key=lambda d: d['cx'] - d['w'] / 2)
        start_time = candidates[0]['text']
        if start_time.find('至') != -1 and len(start_time) > 13:
            start_time, end_time = start_time.split('至')
        else:
            # del candidates[:1]
            if len(candidates) > 1:
                end_time = candidates[1]['text']
                if len(end_time) < 5:
                    tmp_index = get_min_distance_index(candidates[-1],None,candidates)
                    if end_time[-1] == '-' or candidates[tmp_index]['text'][0] == '-':
                        end_time += candidates[tmp_index]['text']
                    else:
                        end_time += '-' + candidates[tmp_index]['text']

    start_time = re.sub(r'Z','2',start_time)
    start_time = re.sub(r'Q', '0', start_time)
    start_time = re.sub(r'T', '1', start_time)

    end_time = re.sub(r'Z','2',end_time)
    end_time = re.sub(r'Q', '0', end_time)
    end_time = re.sub(r'T', '1', end_time)

    start_time = re.sub('至','',start_time)
    end_time = re.sub('至','',end_time)

    # date1 = start_time.split('-')
    # date2 = end_time.split('-')


    if len(start_time.split('-')) == 3 and len(end_time.split('-') )== 3: # 6/10/forever

        # year------
        date1 = start_time.split('-')
        date2 = end_time.split('-')
        year1 = date1[0]
        year2 = date2[0]
    else:
        year1 = start_time[:4]
        year2 = end_time[:4]


    if year1[1] == '8' or year1[1] == '6':
        year1 = year1[0] + '0' + year1[2:]
    if year2[1] == '8' or year2[1] == '6':
        year2 = year2[0] + '0' + year2[2:]

    if year1[2] == '7':
        year1 = year1[:2] + '1' +year1[3:]

    if year1[2] == '8' and year1[:2] == '20':
        year1 = '200' + year1[-1]
    if year2[2] == '8' and year2[:2] == '20':
        year2 = '200' + year2[-1]
    if year1[-1] == '8': # 最后一位矫正不了
        if int(year2) - int(year1) == 6 or int(year2) - int(year1) == 10:
            pass
        else:
            year1 = year1[:3] + '0'
    # ----------

    # month-----
    if len(start_time.split('-')) == 3:
        month = start_time.split('-')[1]
        day = start_time.split('-')[2]
    if len(end_time.split('-')) == 3:
        month = end_time.split('-')[1]
        day = end_time.split('-')[2]
    # month = date1[1]
    if len(month) != 2:
        month = date2[1]
    if month[0] == '8' or month[0] == '6':
        month = '0' + month[1]
    if month[1] == '8' or month[1] == '6':
        if month[0] != '0':
            month = month[0] + '0'
    # ----------

    # day------- 最后一位矫正不了
    # day = date1[2]
    if len(day) != 2:
        day = date2[2]
    if day[0] == '8':
        day ='0' + day[1]
    # ----------
    start_time = year1 + '-' + month + '-' + day
    end_time = year2 + '-' + month + '-' +day
    return start_time, end_time

def getDateOfFirstIssue(data):
    dateFirstIssue = ''

    try :
        candidates = [d for d in data if re.search(r'\d{4}-|^(19|20|28)|-..-|.?..-..|初|次|领|证', d['text']) and len(d['text']) <= 13]
        if len(candidates):
            candidates = sorted(candidates, key=lambda d: d['cy'] - d['h'] / 2)
            tmp_candidate = candidates[0]
            number_list = []
            for i in range(0,len(candidates)):
                tmp = calc_axis_iou(tmp_candidate,candidates[i],1)
                if calc_axis_iou(tmp_candidate,candidates[i],1) > 0.3:
                    number_list.append(candidates[i])
            candidates = [d for d in number_list if re.search(r'\d{4}-|^(19|20|28)|-..-|.?..-..|-..', d['text']) and len(d['text']) <= 13]
            if len(candidates):
                candidates = sorted(candidates, key=lambda d:d['cx'] - d['w'] / 2)
                for i in range(0,len(candidates)):
                    dateFirstIssue += candidates[i]['text']
        year = dateFirstIssue[:4]
        if year[1] == '8':
            year = year[0] + '0' + year[2:]
        if year[2] == '8':
            if year[:2] != '19':
                year = year[:2] + '0' + year[-1]
        dateFirstIssue = year + dateFirstIssue[4:]
    except Exception as e:
        print("getDateOfFirstIssue flase ")
    return dateFirstIssue

def getMaxRight(data):
    maxRight = -1
    for i in range(0,len(data)):
        if data[i]['cx'] + data[i]['w'] / 2 > maxRight:
            maxRight = data[i]['cx'] + data[i]['w'] / 2
    return maxRight

def getCreditCode(data):
    creditCode = ''
    index = 0
    indexes = [i+2 for i, d in enumerate(data[2:]) if re.search(r'[0-9A-Z]{13,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        creditCode = data[indexes[0]]['text']
    creditCode = creditCode.split('号')[-1]
    creditCode = creditCode.split('码')[-1]
    return creditCode,index

def getName(data,index):
    name = ''
    tmp_index = index + 1

    indexes = [i + tmp_index for i, r in enumerate(data[index+1:]) if re.search(r'(名|称)', r['text'])]
    if len(indexes):
        index = indexes[0]
        candidates = []
        candidates.append(data[index])
        for i in range(0,len(data)):
            if calc_axis_iou(data[i],data[index],1) > 0.3 and data[i] not in candidates:
                candidates.append(data[i])
        candidates = sorted(candidates, key=lambda d:len(d['text']))
        name = candidates[-1]['text']
        tmp_index = data.index(candidates[-1])
    return name, tmp_index

def getType(data, index):
    ty = ''
    tmp_index = index + 1
    """
    有限责任公司 股份有限责任公司 个人独资企业 合伙企业 个体工商户
    """
    indexes = [i + tmp_index for i, r in enumerate(data[index+1:]) if re.search(r'(有限|责?任|公司|独资?企业|合伙企业|个体工商户)', r['text'])]
    if len(indexes):
        ty = data[indexes[0]]['text']
        tmp_index = indexes[0]
    else:
        candidates = [d for i,d in enumerate(data[index+1:]) if re.search(r'类|型',d['text'])][:2]
        if len(candidates):
            candidates = sorted(candidates,key=lambda d:d['cx'])
            for i in range(0,len(data)):
                if calc_axis_iou(data[i],candidates[-1],1) > 0.3 and data[i]['cx'] > candidates[-1]['cx']:
                    ty = data[i]
                    break
            tmp_index = data.index(ty)
            ty = ty['text']
    return ty, tmp_index

def getAddress(data,index):

    addr = ''
    tmp_index = index + 1
    indexes = [i + tmp_index for i, d in enumerate(data[index + 1:]) if re.search(r'[\u4e00-\u9fa5]{6,}', d['text'])]
    if len(indexes):
        addr = data[indexes[0]]['text']
        tmp_index = indexes[0]
        for i in range(tmp_index,len(data)):
            if math.fabs((data[indexes[0]]['cy'] + data[indexes[0]]['h'] / 2) - (data[i]['cy'] - data[i]['h'] / 2)) < 10 and math.fabs((data[indexes[0]]['cx'] + data[indexes[0]]['w'] / 2) - (data[i]['cx'] - data[i]['w'] / 2)) < 10:
                addr += data[i]['text']
                tmp_index = i
                break
    return addr, tmp_index

def getLegalPerson(data,index):
    legalPerson = ''
    tmp_index = index + 1
    indexes = [i + tmp_index for i, r in enumerate(data[index+1:]) if re.search(r'(法定|代表人)', r['text'])]

    if len(indexes):
        candidate = []
        for i in range(0,len(data)):
            x_cha = (data[i]['cx'] - data[i]['w'] / 2) - (data[indexes[0]]['cx'] + data[indexes[0]]['w'] / 2)
            if i != indexes[0] and calc_axis_iou(data[i],data[indexes[0]],1) > 0.4 and (data[i]['cx'] - data[i]['w'] / 2) - (data[indexes[0]]['cx'] + data[indexes[0]]['w'] / 2) < 200:
                candidate.append(data[i])
        candidate = sorted(candidate,key=lambda d: d['cx'])
        legalPerson = candidate[-1]['text']
        tmp_index = data.index(candidate[-1])

    return legalPerson, tmp_index

def getRegisterCapital(data,index):
    registerCapital = ''
    tmp_index = index + 1
    indexes = [i + tmp_index for i, r in enumerate(data[index + 1:]) if re.search(r'注册|资本|出资总额', r['text'])]

    if len(indexes):
        candidate = []
        for i in range(0, len(data)):
            if calc_axis_iou(data[i], data[indexes[0]], 1) > 0.3:
                candidate.append(data[i])
        candidate = sorted(candidate, key=lambda d: d['cx'])
        registerCapital = candidate[-1]['text']
        tmp_index = data.index(candidate[-1])
    return registerCapital, tmp_index

def getEstDate(data,index):
    estdate = ''
    tmp_index = index + 1
    candidates = [d for d in data[index+1:] if re.search(r'\d{4}-|^(19|20|28)|年.?.?月.?.?日', d['text'])]
    if len(candidates):
        estdate = candidates[0]['text']
        tmp_index = data.index(candidates[0])
    return estdate, tmp_index

def getBusiTerm(data,index):
    busiTerm = ''
    indexes = [i+index+1 for i, r in enumerate(data[index + 1:]) if re.search(r'\d{4}-|^(19|20|28)|(年.?.?月.?.?日)$', r['text']) and len(r['text']) > 5]
    if len(indexes):
        start_time = ''
        end_time = ''
        candidates = []
        for i in range(0,len(data)):
            if i in indexes:
                candidates.append(data[i])
        if len(candidates):
            candidates = sorted(candidates, key=lambda d: d['cy'])[:2]
            candidates = sorted(candidates, key=lambda d: d['cx'])
            start_time = candidates[0]['text']
            if start_time.find('至') != -1 and len(start_time) > 13:
                start_time, end_time = start_time.split('至')
            else:
                # del candidates[:1]
                if len(candidates) > 1:
                    end_time = candidates[-1]['text']
                    if len(end_time) < 5:
                        tmp_index = get_min_distance_index(candidates[-1], None, candidates)
                        if end_time[-1] == '-' or candidates[tmp_index]['text'][0] == '-':
                            end_time += candidates[tmp_index]['text']
                        else:
                            end_time += '-' + candidates[tmp_index]['text']
            start_time = re.sub('至','',start_time)
            end_time = re.sub('至','',end_time)
            busiTerm = start_time + ' 至 ' + end_time
    else:
        candidates = [d for i,d in enumerate(data[index + 1:]) if re.search(r'长期',d['text'])]
        if len(candidates):
            busiTerm = '长期'
    return busiTerm

def getBusiScope(data,index):

    busiScope = ''

    candidates = [d for i, d in enumerate(data[index+1:]) if re.search(r'范|围', d['text'])][:2]

    if len(candidates):
        scopes = []
        candidates = sorted(candidates, key=lambda d:d['cx'],reverse=True)
        for i in range(index + 1,len(data)):
            if calc_axis_iou(candidates[0],data[i],1) > 0.2 and data[i]['cx'] > candidates[0]['cx']:
                scopes.append(data[i])

        scopes = sorted(scopes,key=lambda d: math.pow((d['cx'] - d['w'] / 2),2) + math.pow((d['cy'] - d['h'] / 2),2))

        while True:
            scp = scopes[0]
            scopes = []
            scopes.append(scp)
            index = data.index(scp)
            for i in range(index-4,len(data)):

              if calc_axis_iou(data[i],scp,1) > 0.3 and data[i]['cx'] >= scp['cx'] and data[i] not in scopes:
                  scopes.append(data[i])
            if len(scopes) == 0:
                break
            scopes = sorted(scopes, key=lambda d: d['cx'])
            for i in range(0,len(scopes)):
                busiScope += scopes[i]['text']
            scp = scopes[0]
            scopes = []
            index = data.index(scp)
            for i in range(index+1,len(data)):
                if calc_axis_iou(scp,data[i]) > 0.01 and math.fabs(data[i]['cy'] - scp['cy']) < 70:
                    scopes.append(data[i])

            if len(scopes) == 0:
                break
            scopes = sorted(scopes, key=lambda d: math.pow((d['cx'] - d['w'] / 2),2) + math.pow((d['cy'] - d['h'] / 2),2))
    return busiScope

def formatResult(data):
    maxRight = getMaxRight(data)

    creditCode,index = getCreditCode(data)
    name, index = getName(data,index)
    ty, index = getType(data,index)
    addr, index = getAddress(data,index)
    legalPerson, index = getLegalPerson(data,index)
    registerCapital, index = getRegisterCapital(data,index)
    estdate,index = getEstDate(data,index)
    busiTerm = getBusiTerm(data,index)
    busiScope = getBusiScope(data,index)


    res = [{'title': r'营业执照信息',
            'items': [[{'name': r'统一社会信用代码', 'value': creditCode},
                       {'name': r'名称', 'value': name},
                       {'name': r'注册资本', 'value': registerCapital},
                       {'name': r'类型', 'value': ty},
                       {'name': r'成立日期', 'value': estdate},
                       {'name': r'法定代表人', 'value': legalPerson},
                       {'name': r'营业期限', 'value': busiTerm},
                       {'name': r'经营范围', 'value': busiScope},
                       {'name': r'住所', 'value': addr}
                       ]]
            }]

    return res
if __name__ == '__main__':

    test()
