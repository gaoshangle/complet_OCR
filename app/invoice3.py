import os
import cv2
import sys
import time
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
from crnn import crnnRec

from eval.invoice.eval_invoice import evaluate
# from layout.VGGLocalization import VGGLoc, trans_image

# python pse
# from pypse import pse as pypse
from pse2 import pse2

def get_params():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/ctw1500_res50_pretrain_ic17.pth.tar',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=0.5,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=2240,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.5,
                        help='min score')
    parser.add_argument('--image_fgbg', nargs='?', type=bool, default=False,
                       help='split image into foreground and background')
    parser.add_argument('--evaluate', nargs='?', type=bool, default=True,
                        help='evalution')
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
    precisions = []
    for idx, (org_img, img) in enumerate(test_loader):
        print('progress: %d / %d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

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

        result = crnnRec(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB), rects)
        result = formatResult(result)

        if args.evaluate:
            image_file = image_name + '.txt'
            error_file = image_name + '-errors.txt'
            file = os.path.join(os.path.dirname(__file__), '../data/gt/', image_file)
            errfile = os.path.join(os.path.dirname(__file__), '../data/error/', error_file)
            if os.path.exists(file):
                precision = evaluate(file, errfile, result)
                print('precision:' + str(precision) + '%')
                precisions.append(precision)

    if len(precisions):
        mean = np.mean(precisions)
        print('mean precision:' + str(mean) + '%')

    # cmd = 'cd %s;zip -j %s %s/*' % ('./outputs/', 'submit_invoice.zip', 'submit_invoice')
    # print(cmd)
    # sys.stdout.flush()
    # util.cmd.Cmd(cmd)
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
    return re.search(r'(货物|劳务|服.?名称|规.?型号|^单[位价]|^数|.?额$|^税.$|项目|类型|车牌|通行日期)', text) != None
    # return text.find(r'服务名称') > -1 \
    #        or text.find(r'规格型号') > -1 \
    #        or text.find(r'单位') > -1 \
    #        or text.find(r'数量') > -1 \
    #        or text.find(r'单价') > -1 \
    #        or text.find(r'金额') > -1 \
    #        or text.find(r'税率') > -1 \
    #        or text.find(r'税额') > -1

def isSummary(text):
    pattern = re.compile(r'[￥|Y|羊]\d+?\.?\d*')
    return text==r'计' or text == r'合' or text == r'合计' or pattern.search(text) != None

def get_content_boundary(data):
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

    # rt = [d for d in data[start:end] if re.match(r'\d+\.?\d*$', d['text'])]
    rt = max(data[start:end], key=lambda x: x['cx']+x['w']/2)

    left = lf['cx']-lf['w']/2
    right = rt['cx']+rt['w']/2

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
    copyed = preprocess_line(copyed)

    if isvehicle:
        # ratio, price = get_ratio(copyed, xmax)
        ratio, price, tax = get_ratio_price_tax(copyed, xmax)
        # title
        title = get_title(copyed, xmin)
        sdate, edate, price, ratio = get_date(copyed, price, ratio)
        platenum, cartype, sdate, edate = get_platenum_cartype(copyed, sdate, edate)

        return postprocess_line(title, platenum, cartype, sdate, edate, price, ratio, tax)
    else:
        # ratio
        # ratio, price = get_ratio(copyed, xmax)
        ratio, price, tax = get_ratio_price_tax(copyed, xmax)
        # title
        title = get_title(copyed, xmin)
        # tax
        # tax = get_tax(copyed, xmax, ratio)
        # prices
        specs, amount, uprice, price, ratio = get_numbers(copyed, price, ratio)
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
    if tax != '***' and (not tax or not is_number(tax) or float(tax) > float(price) or not '.' in tax) and price and ratio:
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
            m = re.match(r'(\d{1,2})\D+', d['text'])
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

def get_numbers(line, price, ratio):
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
                amount = num

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
    dx = abs(data['cx'] - data['w'] / 2 - xmin)
    text = data['text']

    if dx < LEFT_MARGIN and text[0] != '*':
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
    indexes = [i for i, d in enumerate(data) if re.search(r'开.?日期|票日期|校.?码', d['text'])]
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
    candidates = [d for d in basic if re.search(r'^校.?码*', d['text'])]
    if len(candidates):
        m = re.match(r'^校.?码.*?(\d+)', candidates[0]['text'])
        checkcode = m.group(1) if m else ''

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
                    if re.match(r'[\u4e00-\u9fa5]{2,3}', t['text']):
                        title = t['text'] + suffix
                        break

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
    candidates = [d for d in basic if re.search(r'开?票?日期', d['text'])]
    if len(candidates):
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

    xmin = boundary[0]

    rear = data[-10:]
    indexes = [i for i,d in enumerate(rear) if re.search(r'收款|开票|复核|人:|复.:', d['text'])]
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

            s = max([i for i, d in enumerate(data) if re.search(r'\(?大写\)?', d['text'])])
            e = min([i for i, d in enumerate(data) if re.search(r'\(?小写\)?.*(￥|Y|羊)?\d+', d['text'])])
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

    checkcode = get_basic_checkcode(basic)
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

    return res
    # return {"type":type, "title":title, "code":code, "sn":sn, "date":date, "checkcode":checkcode, "payee":payee, "reviewer":reviewer, "drawer":drawer}

def get_buyer_boundary(data):
    indexes = [i for i, d in enumerate(data) if isTitle(d['text'])]
    end = min(indexes)

    indexes = [i for i, d in enumerate(data) if i < end and re.search(r'(开票日期)|(校.?码)', d['text'])]
    start = max(indexes) + 1

    indexes = [i for i, d in enumerate(data[start:end]) if calc_axis_iou(data[start-1], d, 1) > 0.3]
    if len(indexes):
        start = start + max(indexes) + 1

    return start, end

def get_buyer_name(buyer):
    indexes = [i for i, d in enumerate(buyer) if re.search(r'^称:[\u4e00-\u9fa5]{6,}', d['text'])]
    if len(indexes):
        index = indexes[0]
    else:
        indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{6,}', d['text'])]
        index = indexes[0]
    name = buyer[index]
    text = name['text']
    if text.find(':') >= 0:
        name = text.split(':')[1]
    else:
        name = re.sub(r'^[^\u4e00-\u9fa5]+?', '', text)
        name = re.sub(r'^称', '', name)

    return name, index

def get_buyer_taxnumber(buyer):
    indexes = [i for i, d in enumerate(buyer) if re.search(r'[0-9A-Z]{16,}', d['text'])]
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
            address = text.split(':')[1]
        else:
            address = text

        if not re.search(r'[0-9\-]{11,}$', address):
            indexes = [i for i, d in enumerate(buyer) if re.match(r'\d+$', d['text']) and i > index]
            if len(indexes):
                index = indexes[0]
                address += buyer[index]['text']

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
            indexes = [i for i, d in enumerate(buyer) if re.match(r'\d{12,}$', d['text']) and i > index]
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


def getContent(data):
    res = []
    start,end,left,right = get_content_boundary(data)
    content = data[start+1:end]

    isvehicle = isVehicle(data)
    # top = min(content, key=lambda x:float(x['cy'])-float(x['h']/2))
    # bottom = max(content, key=lambda x: float(x['cy'])+float(x['h']/2))
    # lh = (bottom['cy'] + bottom['h']/2 - top['cy'] + top['h']/2) / 8

    lt = min(content, key=lambda x:float(x['cx'])-float(x['w']/2))
    rb = max(content, key=lambda x: float(x['cx'])+float(x['w']/2))
    left = float(lt['cx'])-float(lt['w']/2)
    right = float(rb['cx'])+float(rb['w']/2)

    line = []
    wraptitle = []

    for idx,ct in enumerate(content):
        deal = False
        iswrap = is_wrapped_title(ct, line, [left,right])
        if not iswrap:
            linelen = len(line)
            if linelen:
                y_ious = []
                for l in line:
                    x_iou = calc_axis_iou(l, ct)
                    y_iou = calc_axis_iou(l, ct, 1)
                    y_ious.append(y_iou)
                    if x_iou > 0.3:
                        deal = True
                        break
                if not deal and np.mean(y_ious) < 0.05:
                    deal = True

            if deal == False:
                line.append(ct)
            else:
                check_title(line, content, idx+1, idx+4, [left,right])
                if len(res):
                    check_wrap_title(res, wraptitle, line)
                parsed = parseLine(line, [left, right], isvehicle)
                res.append(parsed)

                line = [ct]
        else:
            wraptitle.append(ct)

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
    s = max([i for i, d in enumerate(data) if re.search(r'\(?大写\)?', d['text'])])
    e = min([i for i, d in enumerate(data) if re.search(r'\(?小写\)?.*(￥|Y|羊)?\d+', d['text'])])

    start = max([s,e]) + 1
    if abs(s-e) == 1:
        start = start + 1
    end = len(data) - 2

    return start, end

def get_seller_name(buyer):
    name = ''
    index = -1

    indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{6,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        name = buyer[index]
        text = name['text']
        if text.find(':') >= 0:
            name = text.split(':')[1]
        else:
            name = re.sub(r'^[^\u4e00-\u9fa5]+?', '', text)
            name = re.sub(r'^称|你', '', name)

    return name, index

def get_seller_taxnumber(buyer):
    taxnumber = ''
    index = -1

    indexes = [i for i, d in enumerate(buyer) if re.search(r':[0-9A-Z]{16,}|^[0-9A-Z]{16,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        taxnumber = buyer[index]
        text = taxnumber['text']
        if text.find(':') >= 0:
            taxnumber = text.split(':')[1]
        else:
            taxnumber = re.sub(r'^[^0-9A-Z]+?', '', text)

    return taxnumber, index

def get_seller_address(buyer):
    address = ''
    index = -1

    indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{7,}', d['text'])]
    if len(indexes):
        index = indexes[0]
        address = buyer[index]
        text = address['text']
        if text.find(':') >= 0:
            address = text.split(':')[1]
        else:
            address = text
        address = re.sub(r'^地址、?电话', '', address)

        if not re.search(r'[0-9\-]{11,}$', address):
            indexes = [i for i, d in enumerate(buyer) if re.match(r'\d+$', d['text']) and i > index]
            if len(indexes):
                index = indexes[0]
                address += buyer[index]['text']

        for prov in PROVINCE:
            idx = address.find(prov)
            if idx > 0:
                address = address[idx:]
                break

    return address, index

def get_seller_account(buyer):
    account = ''
    indexes = [i for i, d in enumerate(buyer) if re.search(r'[\u4e00-\u9fa5]{7,}', d['text'])]
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
            indexes = [i for i, d in enumerate(buyer) if re.match(r'\d+$', d['text']) and i > index]
            if len(indexes):
                index = indexes[0]
                account += buyer[index]['text']

        idx = account.find(r'账号')
        if idx >= 0:
            account = account[idx+2:]

    return account

def getSeller(data):
    start, end = get_seller_boundary(data)
    seller = data[start:end]

    name, index = get_seller_name(seller)
    seller = seller[index + 1:]

    taxnum, index = get_seller_taxnumber(seller)
    seller = seller[index + 1:]

    address, index = get_seller_address(seller)
    seller = seller[index + 1:]

    account = get_seller_account(seller)

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
    if len(prices):
        prices = sorted(prices, key=lambda x: x['cy'], reverse=True)
        p = prices[0]
        if calc_axis_iou(p, prices[1:], 1) < 0.01:
            total = p
            prices.remove(p)

    if len(prices):
        prices = sorted(prices, key=lambda x: x['cx'], reverse=True)
        p = prices[0]
        if abs(p['cx']+p['w']/2 - right) < RIGHT_MARGIN:
            tax = p
            prices.remove(p)

    if len(prices):
        price = prices[0]

    total = check_price(total, calctotal)
    price = check_price(price, calcprice)
    tax = check_price(tax, calctax)

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

    res = [[{'name': r'金额合计', 'value': price},
            {'name': r'税额合计', 'value': tax},
            {'name': r'价税合计(大写)', 'value': capital},
            {'name': r'价税合计(小写)', 'value': total}]]
    return res

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
