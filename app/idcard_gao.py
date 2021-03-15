# -*- coding:utf-8 -*-
import math
import util
from apphelper.image import  calc_iou
from crnn import crnnRec
# from angle import *

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
    for i in range(0,len(near)):
        x_tem = float(near[i]['cx']) - float(near[i]['w']/2)
        y_tem = float(near[i]['cy']) + float(near[i]['h']/2)
        distance_tem = math.sqrt(math.pow((x-x_tem),2) + math.pow((y-y_tem),2))
        if distance_tem < distance_min and len(near[i]['text']) and calc_axis_iou(near[i],tem,1) > 0.3 and float(tem['cx']) < float(near[i]['cx']):
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
    parser.add_argument('--model_path', default=r'/home/share/gaoluoluo/模型/model50_latest_epoch152.pth', type=str)
    # parser.add_argument('--input_folder', default='/home/share/gaoluoluo/invoices_gao_true/img/', type=str, help='img path for predict')
    parser.add_argument('--input_folder', default='/home/share/gaoluoluo/complete_IDCard/input/', type=str, help='img path for predict')
    parser.add_argument('--img_correct',default='/home/share/gaoluoluo/dbnet/test/test_corre_input/',type=str,help='img_correct path for predict')
    # parser.add_argument('--input_folder',default='/home/share/gaoluoluo/dbnet/test/test_corre_input', type=str, help='img path for predict')
    # parser.add_argument('--input_folder', default='/home/share/gaoluoluo/complete_ocr/data/images/tmp', type=str,help='img path for predict')
    parser.add_argument('--output_folder', default='/home/share/gaoluoluo/dbnet/test/test_output/', type=str, help='img path for output')
    # parser.add_argument('--output_folder', default='/home/share/gaoluoluo/invoices_gao_true/outputs/', type=str, help='img path for output')
    parser.add_argument('--gt_txt', default='/home/share/gaoluoluo/complete_ocr/data/txt_not_use', type=str, help='img 对应的 txt')
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
    total_frame = 0.0
    total_time = []

    names = []
    for ind in range(0,1):  # img_path /home/share/gaoluoluo/dbnet/test/input/2018实验仪器发票.jpg
        # print("img_path:",img_path)  /home/share/gaoluoluo/dbnet/test/input/2018实验仪器发票.jpg
        img_path = tmpfile
        print("\nimg_path:", img_path)
        # img_path = '/home/share/gaoluoluo/IDCard/corre_id/3.jpg'
        start_time = time.time()
        img = angle_corre(img_path)# 调整图片角度
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
            # points = [[0, 0], [1, 0], [1, 1], [0, 1]]
            # points = np.array(points, np.float32)
            # print(cv2.minAreaRect(points))
            # print(type(points))
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
        result = crnnRec(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB), rects)
        # result = crnnRec_tmp(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB), rects)
        result = list(reversed(result))

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
        print("--------------")

        # print("-------------------11")
        result = formatResult(result)


    return result
import re

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

def getName(data):
    indexes = [i for i, d in enumerate(data) if re.search(r'^姓[\u4e00-\u9fa5]{1,}', d['text'])]
    index = 0
    if len(indexes):
        max_cal = 0
        for i in range(0,len(data)):
            if i != indexes[0]:
                tmp_cal = calc_axis_iou(data[indexes[0]],data[i],1)
                if tmp_cal > max_cal:
                    max_cal = tmp_cal
                    index = i
    else:
        indexes = [i for i, d in enumerate(data) if re.search(r'[\u4e00-\u9fa5]{2,}', d['text'])]
        index = indexes[0]
    name = data[index]
    name = name['text']
    name = name.split('姓名',1)[-1]
    # name = re.sub(r, '', name)
    return name, index
NATIONS = ['汉', '满', '蒙古', '回', '藏', '维吾尔', '苗', '彝', '壮', '布依', '侗', '瑶', '白', '土家', '哈尼', '哈萨克', '傣', '黎', '傈僳', '佤', '畲', '高山', '拉祜', '水', '东乡', '纳西', '景颇', '柯尔克孜', '土', '达斡尔', '仫佬', '羌', '布朗', '撒拉', '毛南', '仡佬', '锡伯', '阿昌', '普米', '朝鲜', '塔吉克', '怒', '乌孜别克', '俄罗斯', '鄂温克', '德昂', '保安', '裕固', '京', '塔塔尔', '独龙', '鄂伦春', '赫哲', '门巴', '珞巴', '基诺']
def getNation(data):

    sex = ''
    nation = ''
    indexes = [i for i, d in enumerate(data) if re.search(r'性别|民族', d['text']) ]
    index = 0
    if len(indexes):
        sex_nation_list = []
        sex_nation_list.append(data[indexes[0]])
        for i in range(0, len(data)):
            if i != indexes[0]:
                tmp_cal = calc_axis_iou(data[indexes[0]], data[i], 1)
                if tmp_cal > 0.5 and data[i] not in sex_nation_list:
                    sex_nation_list.append(data[i])
        max_x = 0
        for i in range(0,len(sex_nation_list)):
            if sex_nation_list[i]['cx'] > max_x:
                max_x = sex_nation_list[i]['cx']
                nation = sex_nation_list[i]
        sex = [d for i, d in enumerate(sex_nation_list) if re.search(r'男|女', d['text'])]
        if len(sex):
            sex = sex[0]
    else:
        indexes = [i for i, d in enumerate(data) if re.search(r'男', d['text'])]
        if len(indexes):
            sex = '男'
        else:
            sex = '女'
        nation = '汉'
        return sex,nation

    if sex != '' and sex != []:
        sex = sex['text']
        sex = sex.split('性',1)[-1]
        sex = sex.split('别', 1)[-1]
        # sex = re.sub(r'^', '', sex)
    if nation != '':
        nation = nation['text']
        nation = nation.split('民',1)[-1]
        nation = nation.split('族', 1)[-1]
        for i in range(0,len(NATIONS)):
            if NATIONS[i] in nation:
                nation = NATIONS[i]
                break
        # nation = re.sub(r'民族','',nation)
    return sex, nation

def filterChinese(date):

    year = ''
    month = ''
    day = ''

    flag = 0
    flag2 = 0
    for i in range(0,len(date)):
        if re.search(r'\d{1}',date[i]) and len(year) < 4:
            year += date[i]
            continue
        if re.search(r'\d{1}',date[i]) and len(year) == 4 and flag == 0 and flag2 == 1:
            month += date[i]
            continue

        if flag == 1 and re.search(r'\d{1}',date[i]):
            day += date[i]

        if len(year) == 4 and len(month) and re.search(r'[\u4e00-\u9fa5]{1}',date[i]):
            flag = 1
        if len(year) == 4 and re.search(r'[\u4e00-\u9fa5]{1}',date[i]):
            flag2 = 1
    if len(year)==4 and len(month) and len(day):
        return year + '年' + month + '月' + day + '日'
    else:
        return date

def getBrithday(data):
    candidates = [d for d in data if re.search(r'\d{4}|^(19|20).?.?年?|^出生?|.月|.日', d['text'])]
    candidates = sorted(candidates, key=lambda d: d['cx'] - d['w'] / 2)
    if len(candidates):# 原来的Pse wangluo shi zhengti kuangde
        min_x = 9999
        for i in range(0, len(candidates)):
            if candidates[i]['cx'] < min_x:
                min_x = candidates[i]['cx']
                year = candidates[i]

        tmp_year= re.sub(r'出','',year['text'])
        tmp_year= re.sub(r'生','',tmp_year)
        if len(tmp_year) > 6:
            date = tmp_year
        else:
            index = get_min_distance_index(year,None,candidates)
            date = tmp_year + candidates[index]['text']
            if len(date) < 6:
                index = get_min_distance_index(candidates[index],None,candidates)
                date = date + candidates[index]['text']

    date = re.sub(r'出生','',date)
    date = filterChinese(date)
    return date
def getAddress(data):
    indexes = [i for i, d in enumerate(data) if re.search(r'[\u4e00-\u9fa5]{8,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        tmp_x = data[index]['cx'] - data[index]['w'] / 2
        address = data[index]['text']
        for _,da in enumerate(data[index + 1:]):
            xx = math.fabs(int(da['cx'] - da['w'] / 2) - int(tmp_x))
            # get_min_distance_index()
            if math.fabs(int(da['cx'] - da['w'] / 2) - int(tmp_x)) < 20:
                address += da['text']
    return address
def getIDCode(data):
    indexes = [i for i, d in enumerate(data) if re.search(r'\d{10,}', d['text'])]
    code = data[indexes[0]]['text']
    code = re.sub(r'公民身份号码', '', code)
    return code

def formatResult(data):

    name, index = getName(data[:5])
    sex, nation = getNation(data[index+1:])
    brithday = getBrithday(data)
    addr = getAddress(data)
    IDCode = getIDCode(data)

    # res = [{'title':r'姓名', 'items':name},
    #        {'title':r'性别', 'items':sex},
    #        {'title':r'民族', 'items':nation},
    #        {'title': r'出生', 'items':brithday},
    #        {'title':r'住址', 'items':addr},
    #        {'title':r'公民身份号码', 'items':IDCode}]

    if name:
        title = '正面信息'
        res = [{'title':title,
                'items':[[{'name':r'姓名', 'value':name},
                          {'name':r'性别', 'value':sex},
                          {'name':r'民族', 'value':nation},
                          {'name':r'出生', 'value':brithday},
                          {'name':r'住址', 'value':addr},
                          {'name':r'公民身份号码', 'value':IDCode}]]}]
    else:
        title = '反面信息'
        res = [{'title': title,
                'items': [[{'name': r'签发机关', 'value': org},
                           {'name': r'有效期限', 'value': expire}]]}]
    return res
if __name__ == '__main__':

    test()
