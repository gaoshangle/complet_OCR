import os
import cv2
import sys
import re
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

# python pse
# from pypse import pse as pypse
from pse2 import pse2

def get_params():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='mobilenet')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/psenet_lite_mbv2.pth',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=0.5,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=6,
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
        iou = min(ious)
    elif isinstance(a, list):
        if axis == 0:
            ious = [calc_iou([x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2]) for x in a]
        else:
            ious = [calc_iou([x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2]) for x in a]
        iou = min(ious)
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
            if calc_axis_iou(d, row, 1) < 0.05:
                row = mergeRow(row)
                rows.append(row)
                row = []
        row.append(d)

    if len(row):
        row = mergeRow(row)
        rows.append(row)
    return rows

def get_socialcode(rows):
    socialcode = ''

    for idx,rw in enumerate(rows):
        d = [r for r in rw if re.match(r'[A-Z0-9]{15,20}', r['text'])]
        if len(d):
            socialcode = d[0]['text']
            break

    return socialcode

def get_companyname(rows):
    cname = ''
    for idx, rw in enumerate(rows):
        i = [i for i,r in enumerate(rw) if re.match(r'(名|称)', r['text'])]
        if len(i):
            i = max(i)
            cname = rw[i+1]['text']
            break

    return cname

def get_regcapital(rows):
    regcapital = ''
    for idx, rw in enumerate(rows):
        i = [i for i,r in enumerate(rw) if r['text'] == r'注册资本']
        if len(i):
            i = max(i)
            regcapital = rw[i+1]['text']
            break

    return regcapital

def get_busitype(rows):
    busitype = ''
    for idx, rw in enumerate(rows):
        i = [i for i, r in enumerate(rw) if re.match(r'(类|型)', r['text'])]
        if len(i):
            i = max(i)
            busitype = rw[i+1]['text']
            break

    return busitype

def get_establishdate(rows):
    estdate = ''
    for idx, rw in enumerate(rows):
        i = [i for i, r in enumerate(rw) if re.match(r'成立日期', r['text'])]
        if len(i):
            i = max(i)
            estdate = rw[i+1]['text']
            break

    return estdate

def get_legalperson(rows):
    legalperson = ''
    for idx, rw in enumerate(rows):
        i = [i for i, r in enumerate(rw) if re.match(r'法定代表人', r['text'])]
        if len(i):
            i = max(i)
            legalperson = rw[i+1]['text']
            break

    return legalperson

def get_busiterm(rows):
    busiterm = ''
    for idx, rw in enumerate(rows):
        i = [i for i, r in enumerate(rw) if r['text'] == r'营业期限']
        if len(i):
            i = max(i)
            busiterm = rw[i+1]['text']
            break

    return busiterm


def get_busiscope(rows):
    busiscope = []
    busiscope_item = None

    pattern = r'登记机关'
    for idx, rw in enumerate(rows):
        if not busiscope_item:
            i = [i for i, r in enumerate(rw) if re.match(r'(经营范围|住|所)', r['text'])]
            if len(i):
                if len(i) == 1:
                    pattern = r'(住|所)'
                    s = i[0]
                    e = s + 1
                else:
                    i = sorted(i)
                    s = i[0]+1
                    e = i[1]
                for d in rw[s:e]:
                    if not busiscope_item:
                        busiscope_item = d
                    else:
                        busiscope_item['w'] = busiscope_item['w'] + d['w']
                        busiscope_item['cx'] = (busiscope_item['cx'] + d['cx']) / 2
                        busiscope_item['text'] = busiscope_item['text'] + d['text']
                busiscope.append(busiscope_item['text'])
        else:
            i = [i for i, r in enumerate(rw) if re.match(pattern, r['text'])]
            if not len(i):
                for r in rw:
                    if calc_axis_iou(busiscope_item, r) > 0.1:
                        busiscope.append(r['text'])
            else:
                break

    busiscope = ''.join(busiscope)
    return busiscope

def get_addr(rows):
    addr = []
    addr_item = None

    pattern = r'登记机关'
    for idx, rw in enumerate(rows):
        if not addr_item:
            i = [i for i, r in enumerate(rw) if re.match(r'(住|所)', r['text'])]
            if len(i):
                i = max(i)
                addr_item = rw[i+1]
                addr.append(addr_item['text'])
        else:
            i = [i for i, r in enumerate(rw) if re.match(pattern, r['text'])]
            if not len(i):
                if len(addr) < 2:
                    r = [r for r in rw if calc_axis_iou(addr_item, r) > 0.1]
                    if len(r):
                        addr.append(r[0]['text'])
                        break
            else:
                break

    addr = ''.join(addr)
    return addr

def formatResult(data):
    rows = getRows(data)
    socialcode = get_socialcode(rows)
    companyname = get_companyname(rows)
    regcapital = get_regcapital(rows)
    busitype = get_busitype(rows)
    estdate = get_establishdate(rows)
    legalperson = get_legalperson(rows)
    busiterm = get_busiterm(rows)
    busiscope = get_busiscope(rows)
    addr = get_addr(rows)

    res = [{'title': r'营业执照信息',
            'items': [[{'name': r'统一社会信用代码', 'value': socialcode},
                       {'name': r'名称', 'value': companyname},
                       {'name': r'注册资本', 'value': regcapital},
                       {'name': r'类型', 'value': busitype},
                       {'name': r'成立日期', 'value': estdate},
                       {'name': r'法定代表人', 'value': legalperson},
                       {'name': r'营业期限', 'value': busiterm},
                       {'name': r'经营范围', 'value': busiscope},
                       {'name': r'住所', 'value': addr}
                       ]]
            }]

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='mobilenet')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/psenet_lite_mbv2.pth',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=0.5,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=6,
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

    args = parser.parse_args()
    test(args)
