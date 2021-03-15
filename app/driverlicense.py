import os
import cv2
import sys
import re
import time
import datetime
import math
import copy
from functools import reduce
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torch.autograd import Variable
from torch.utils import data

from dataset import DataLoader
import models
import util
# c++ version pse based on opencv 3+
from pse import pse

# from angle import detect_angle
from apphelper.image import order_point, calc_iou, xy_rotate_box
from crnn_lite import crnnRec

# python pse
# from pypse import pse as pypse
from pse2 import pse2

def get_params():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='mobilenet')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/psenet_lite_mbv2.pth',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=0.3,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=6,
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
    return args

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
            dir = os.path.join(os.path.dirname(__file__), '../data/images/invoice/')
            file = file if file is not None else 'tmp.jpg'
            tmpfile = os.path.join(dir, file)
            im.save(tmpfile)

            data = test(get_params(), tmpfile)
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


def test(args, file=None):
    result = []
    data_loader = DataLoader(long_size=args.long_size, file=file)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    slice = 0
    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "mobilenet":
        model = models.Mobilenet(pretrained=True, num_classes=6, scale=args.scale)
        slice = -1

    for param in model.parameters():
        param.requires_grad = False

    # model = model.cuda()

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value

            try:
                model.load_state_dict(d)
            except:
                model.load_state_dict(checkpoint['state_dict'])

            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()

    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print('progress: %d / %d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        (h,w,c) = org_img.shape
        if w < h:
            args.binary_th = 1

        # torch.cuda.synchronize()
        start = time.time()

        # angle detection
        # org_img, angle = detect_angle(org_img)
        outputs = model(img)

        score = torch.sigmoid(outputs[:, slice, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, slice, :, :]
        kernels = outputs
        # kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy( )[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

        if args.arch == 'mobilenet':
            pred = pse2(kernels, args.min_kernel_area / (args.scale * args.scale))
        else:
            # c++ version pse
            pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
            # python version pse
            # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))

        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
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

            ((cx,cy),(w,h),degree) = cv2.minAreaRect(points)
            rect = ((cx,cy),(w+8,h+8),degree)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            bbox = order_point(bbox)
            # bbox = np.array([bbox[1], bbox[2], bbox[3], bbox[0]])
            bboxes.append(bbox.reshape(-1))

            rec = []
            rec.append(rect[-1])
            rec.append((rect[1][1]) * scale[1])
            rec.append((rect[1][0]) * scale[0])
            rec.append(rect[0][0] * scale[0])
            rec.append(rect[0][1] * scale[1])
            rects.append(rec)

        # torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f' % (total_frame / total_time))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_invoice/')

        text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box]], 'data/images/tmp/')

        # result = crnnRec(binarize(org_img), rects)
        result = crnnRec(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB), rects)
        result = formatResult(result)

    # cmd = 'cd %s;zip -j %s %s/*' % ('./outputs/', 'submit_invoice.zip', 'submit_invoice')
    # print(cmd)
    # sys.stdout.flush()
    # util.cmd.Cmd(cmd)
    return result

def calc_axis_iou(a,b,axis=0):
    if isinstance(b, list):
        if axis == 0:
            ious = [calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2]) for x in b]
        else:
            ious = [calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2]) for x in b]
        iou = max(ious)
    elif isinstance(a, list):
        if axis == 0:
            ious = [calc_iou([x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2]) for x in a]
        else:
            ious = [calc_iou([x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2]) for x in a]
        iou = max(ious)
    else:
        if axis == 0:
            iou = calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2])
        else:
            iou = calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2])
    return iou

def mergeRow(row):
    row = sorted(row, key=lambda x: x['cx'] - x['w'] / 2)
    res = []

    i = 0
    j = 1
    while i < len(row) and j < len(row):
        x = row[i]
        y = row[j]
        x1 = x['cx'] + x['w'] / 2
        y1 = y['cx'] - y['w'] / 2

        if abs(x1 - y1) < 8:
            x['w'] = y['cx'] + y['w']/2 - x['cx'] + x['w']/2
            x['cx'] = (y['cx'] + x['cx']) / 2
            x['text'] = x['text'] + y['text']
            j = j + 1
        else:
            res.append(x)
            i = j
            j = j + 1

    res.append(row[i])
    return res


def getRows(data):
    data = sorted(data, key=lambda x: x['cy']-x['h']/2)
    rows = []
    row = []
    for d in data:
        if len(row):
            if calc_axis_iou(d, row, 1) < 0.2:
                # row = mergeRow(row)
                row = sorted(row, key=lambda x: x['cx'])
                rows.append(row)
                row = []
        row.append(d)

    if len(row):
        # row = mergeRow(row)
        row = sorted(row, key=lambda x: x['cx'])
        rows.append(row)
    return rows

def get_cardno(rows):
    cardno = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'证号', r['text'])]
    #     if len(i):
    #         i = max(i)
    #         r = [r for r in rw[i+1:] if re.match(r'\d{15,}[a-zA-Z]?', r['text'])]
    #         if len(r):
    #             cardno = r[0]['text']
    #         break
    row = rows[2]
    rw = [r['text'] for r in row if re.search(r'\d{15,}[a-zA-Z]?', r['text'])]
    if len(rw):
        rw = rw[0]
        m = re.search(r'(\d{15,}[a-zA-Z]?)', rw)
        cardno = m.group(1)
    return cardno

def get_name(rows):
    name = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'姓名', r['text'])]
    #     if len(i):
    #         i = max(i)
    #         name = rw[i+1]['text']
    #         break
    row = rows[3]
    i = [i for i, r in enumerate(row) if re.search(r'姓名|me', r['text'])]
    if len(i):
        i = max(i)
        names = [r for r in row[i+1:i+3] if re.search('[\u4e00-\u9fa5]{1,}', r['text']) and calc_axis_iou(row[i],r)<0.001]
        if len(names):
            prev = names[-1]
            for n in names:
                if calc_axis_iou(n, prev) > 0:
                    name = name + re.sub(r'[^\u4e00-\u9fa5]', '', n['text'])
                    prev = n
                else:
                    break
    return name

def get_sex(rows):
    sex = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'性别', r['text'])]
    #     if len(i):
    #         i = max(i)
    #         r = [r for r in rw[i + 1:] if re.search(r'(男|女)', r['text'])]
    #         if len(r):
    #             text = r[0]['text']
    #             if text.find(r'男') >= 0:
    #                 sex = r'男'
    #             else:
    #                 sex = r'女'
    #         break
    row = rows[3]
    i = [i for i, r in enumerate(row) if re.search(r'性别|[S|s]ex', r['text'])]
    if len(i):
        i = max(i)
        r = [r['text'] for r in row[i + 1:min(i + 3,len(row))] if re.match('[\u4e00-\u9fa5]{1}', r['text'])]
        if len(r):
            sex = r[0]
    if not sex:
        r = [r for r in row if re.match(r'女|男', r['text'])]
        if len(r):
            text = r[0]['text']
            if text.find(r'女') >= 0:
                sex = r'女'
            else:
                sex = r'男'
    return sex

def get_nationality(rows):
    nation = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'国籍', r['text'])]
    #     if len(i):
    #         i = max(i)
    #         text = rw[i+1]['text']
    #         if len(text) < 2:
    #             r = [r for r in rw[:i] if re.search(r'(男|女)', r['text']) and len(r'text') > 3]
    #             if len(r):
    #                 text = r[-1]['text']
    #                 i = max([text.find(r'男'), text.find(r'女')])
    #                 if i >= 0:
    #                     nation = re.sub(r'[^\u4e00-\u9fa5]', '', text[i+1:])
    #         else:
    #             nation = text
    #         break
    row = rows[3]
    i = [i for i, r in enumerate(row) if re.search(r'国籍|ity', r['text'])]
    if len(i):
        i = max(i)
        r = [r['text'] for r in row[i + 1:min(i + 3, len(row))] if re.search('[\u4e00-\u9fa5]{2,}', r['text'])]
        if len(r):
            nation = re.sub(r'[^\u4e00-\u9fa5]', '', r[0])
    return nation

def get_addr(rows):
    addr = ''
    # s = 0
    # e = 0
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'住址', r['text'])]
    #     if len(i):
    #         s = idx
    #         continue
    #     i = [i for i, r in enumerate(rw) if re.match(r'出生日期', r['text'])]
    #     if len(i):
    #         e = idx
    #
    #     if s and e:
    #         for rw in rows[s:e]:
    #             addr = addr + ''.join([r['text'] for r in rw if r['text'].find(r'住址') < 0])
    #         break
    row = rows[4]
    r = [r for r in row if re.match('[\u4e00-\u9fa5]{5,}', r['text'])]
    if len(r):
        addr = r[0]['text']
        next = rows[5]
        nxt = [n for n in next if re.search(r'\d{4,}[\-\.\d]+', n['text'])]
        if not len(nxt):
            addr = addr + ''.join([n['text'] for n in next if calc_axis_iou(r[0],n)>0.1])
            rows.pop(5)

    return addr

def get_birthday(rows):
    birthday = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'出生日期', r['text'])]
    #     if len(i):
    #         i = max(i)
    #         birthday = rw[i+1]['text']
    #         birthday = validate_date(birthday)
    #         break
    row = rows[5]
    r = [r for r in row if re.search(r'\d{4,}[\-\.\d]+', r['text'])]
    if len(r):
        birthday = r[0]['text']
        birthday = validate_date(birthday)

    return birthday

def get_issuedate(rows):
    issuedate = ''
    row = rows[6]
    r = [r for r in row if re.search(r'\d{4,}[\-\.\d]+', r['text'])]
    if len(r):
        issuedate = r[0]['text']
        issuedate = validate_date(issuedate)

    return issuedate

def get_class(rows):
    cls = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'准驾车型', r['text'])]
    #     if len(i):
    #         cls = rw[-1]['text']
    #         break
    row = rows[7]
    text = row[-1]['text']
    cls = text if len(text) <= 2 else ''
    if cls:
        letter = cls[-1]
        if letter == 'L' or letter == 'I':
            cls = cls[:-1] + '1'
    return cls

def get_validperiod(rows):
    validperiod = ''
    # for idx, rw in enumerate(rows):
    #     i = [i for i, r in enumerate(rw) if re.match(r'有效期限', r['text'])]
    #     if len(i):
    #         validperiod = rw[-1]['text']
    #         dates = validperiod.split(r'至')
    #         if len(dates) > 1:
    #             start = validate_date(dates[0])
    #             end = str(int(start[0:4])+6) + start[4:]
    #             validperiod = start + r'至' + end
    #         break
    row = rows[8]
    rw = [r for r in row if re.search(r'\d{4,}[\-\.\d]+', r['text'])]
    if len(rw):
        dates = []
        for r in rw:
            dt = re.findall(r'\d{4,}[\-\.\d]+', r['text'], re.I)
            dates.extend([validate_date(d) for d in dt])
        validperiod = r'至'.join(dates)
    return validperiod

def validate_date(date):
    date = re.sub(r'[^0-9\-]' ,'', date)
    dates = date.split('-')
    l = len(dates)
    if l == 1:
        date = dates[0]
        year = date[0:4]
        if len(date[4:]) == 4:
            mon = date[4:6]
            day = date[6:]
        else:
            day = date[-2:]
            mon = date[-4:-2]
    elif l == 2:
        year = dates[0]
        if len(dates[1]) == 4:
            mon = dates[1][0:2]
            day = dates[1][2:]
        else:
            day = dates[1][-2:]
            mon = dates[1][-4:-2]
    elif l == 3:
        year = dates[0]
        mon = dates[1]
        day = dates[2]

    now = datetime.datetime.now()
    if int(year) > now.year:
        year = correct_date(year, now.year)
    if int(mon) > 12:
        mon = correct_date(mon, 12)
    if int(day) > 31:
        day = correct_date(day, 31)

    date = year + '-' + mon + '-' + day
    return date

def correct_date(date, thresh):
    date = list(date)
    for i in range(0, len(date)):
        if int(''.join(date)) <= thresh:
            break
        date[i] = correct_digit(date[i])
    date = ''.join(date)
    return date

def correct_digit(digit):
    if digit == '8':
        digit = '0'
    elif digit == '9':
        digit = '0'
    elif digit == '6':
        digit = '0'
    elif digit == '3':
        digit = '2'
    return digit

from skimage.filters import threshold_sauvola
def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    thresh_sauvola = threshold_sauvola(gray, window_size=25)
    binary_sauvola = gray > thresh_sauvola

    binary_sauvola.dtype = 'uint8'
    binary_sauvola = binary_sauvola * 255
    blur = cv2.GaussianBlur(binary_sauvola, (3, 3), 1)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=300, maxLineGap=10)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    # cv2.imshow("line_detect_possible_demo", image)
    return blur

def formatResult(data):
    rows = getRows(data)
    cardno = get_cardno(rows)
    name = get_name(rows)
    sex = get_sex(rows)
    nation = get_nationality(rows)
    addr = get_addr(rows)
    birthday = get_birthday(rows)
    issuedate = get_issuedate(rows)
    cls = get_class(rows)
    validperiod = get_validperiod(rows)

    res = [{'title': r'驾驶证信息',
            'items': [[{'name': r'证号', 'value': cardno},
                       {'name': r'姓名', 'value': name},
                       {'name': r'性别', 'value': sex},
                       {'name': r'国籍', 'value': nation},
                       {'name': r'住址', 'value': addr},
                       {'name': r'出生日期', 'value': birthday},
                       {'name': r'初次领证日期', 'value': issuedate},
                       {'name': r'准驾车型', 'value': cls},
                       {'name': r'有效期限', 'value': validperiod}
                       ]]
            }]
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/ctw1500_res50_pretrain_ic17.pth.tar',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.5,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=1080,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.5,
                        help='min score')

    args = parser.parse_args()
    test(args)
