# coding:utf-8
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
from apphelper.image import order_point, calc_iou, xy_rotate_box, solve
from crnn import crnnRec
from eval.invoice.eval_invoice_old import evaluate
from layout.VGGLocalization import VGGLoc, trans_image
from layout.invoice_layout import detect_layout, get_roi
from app import invoice as EI

# python pse
# from pypse import pse as pypse
from pse2 import pse2

# localization for cross point
vggloc = VGGLoc()
vggloc.load_pretrain("cpu")
vggloc.load_state_dict(torch.load('./checkpoints/layout_model.pth', map_location = "cpu"))


def get_params() :
    parser = argparse.ArgumentParser(description = 'Hyperparams')
    parser.add_argument('--arch', nargs = '?', type = str, default = 'resnet50')
    parser.add_argument('--resume', nargs = '?', type = str,
                        default = './checkpoints/ctw1500_res50_pretrain_ic17.pth.tar',
                        help = 'Path to previous saved model to restart from')
    # parser.add_argument('--resume', nargs = '?', type = str,
    #                     default = '/home/share/gaoluoluo/dbnet/output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth',
    #                     help = 'Path to previous saved model to restart from')

    parser.add_argument('--binary_th', nargs = '?', type = float, default = 0.7,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs = '?', type = int, default = 7,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs = '?', type = int, default = 1,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs = '?', type = int, default = 2240,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs = '?', type = float, default = 5.0,
                        help = 'min kernel area')
    parser.add_argument('--min_area', nargs = '?', type = float, default = 300.0,
                        help = 'min area')
    parser.add_argument('--min_score', nargs = '?', type = float, default = 0.5,
                        help = 'min score')
    parser.add_argument('--evaluate', nargs = '?', type = bool, default = True,
                        help = 'evalution')
    args = parser.parse_args()
    return args


def recognize(im = None, path = None) :
    ret = None
    try :
        file = None
        if im :
            pass
        elif path :
            # 提取文件路径
            # dir,base = os.path.split(path)
            # file,suffix = os.path.splitext(base)
            # dir = os.path.dirname(__file__)
            # tmpfile = os.path.join(dir, 'tmp/'+file+'-large'+suffix)
            # 修改图片大小和分辨率
            im = Image.open(path)
            file = os.path.basename(path)

        if im :
            dir = os.path.join(os.path.dirname(__file__), 'data/images/invoice/')
            file = file if file is not None else 'tmp.jpg'
            tmpfile = os.path.join(dir, file)
            im.save(tmpfile)

            data = test(get_params(), tmpfile)
            if data :
                ret = format(data)
    except Exception as e :
        print(e)

    return ret


def format(data) :
    return data


def extend_3c(img) :
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis = 2)
    return img


def debug(idx, img_paths, imgs, output_root) :

    """

    :param idx:
    :param img_paths:
    :param imgs:
    :param output_root:  data/image/
    :return:
    """
    if not os.path.exists(output_root) :
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)) :
        row = []
        for j in range(len(imgs[i])) :
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis = 1)
        col.append(res)
    res = np.concatenate(col, axis = 0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)


def write_result_as_txt(image_name, bboxes, path) :
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes) :
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def polygon_from_points(points) :
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype = 'int32')
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


def test(args, file = None) :
    result = []
    data_loader = DataLoader(long_size = args.long_size, file = file) # 返回 dataloaderd 对象,为什么还要加载数据
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size = 1,
        shuffle = False,
        num_workers = 2,
        drop_last = True)

    slice = 0
    # Setup Model
    if args.arch == "resnet50" : # resnet50 ?? dbnet 训练使用的是resnet18
        model = models.resnet50(pretrained = True, num_classes = 7, scale = args.scale)
    elif args.arch == "resnet101" :
        model = models.resnet101(pretrained = True, num_classes = 7, scale = args.scale)
    elif args.arch == "resnet152" :
        model = models.resnet152(pretrained = True, num_classes = 7, scale = args.scale)
    elif args.arch == "mobilenet" :
        model = models.Mobilenet(pretrained = True, num_classes = 6, scale = args.scale)
        slice = -1

    # 上面的resnet50已经加载了预训练的一些参数的
    for param in model.parameters() :
        # print("param:",param.name)全部是none
        param.requires_grad = False

    # model = model.cuda( )

    if args.resume is not None :
        if os.path.isfile(args.resume) :
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) # 这个加载的模型是？？

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            print("len(checkpoint['state_dict'].items()):",len(checkpoint['state_dict'].items()))
            for key, value in checkpoint['state_dict'].items() : #获取需要的参数
                # print("key:",key)
                tmp = key[7 :]
                d[tmp] = value
            # print("d:",d)
            try :
                model.load_state_dict(d) # d 是checkpoint 加载的模型参数 model是resnet的
            except :
                model.load_state_dict(checkpoint['state_dict'])

            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else :
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()

    total_frame = 0.0
    total_time = 0.0
    # print("shape(test_loader):",np.shape(test_loader))
    precisions1 = []
    for idx, (org_img, img) in enumerate(test_loader) :
        # print("idx:",idx)
        # print(np.shape(org_img))
        # print("org_img:",org_img)
        # print(np.shape(img))
        # print("img:",img)

        try :
            print('progress: %d / %d' % (idx, len(test_loader)))
            sys.stdout.flush() # 显示地让缓冲区的内容输出

            # img = Variable(img.cuda(), volatile=True)
            org_img = org_img.numpy().astype('uint8')[0]
            # print("org_img:",org_img)
            text_box = org_img.copy()
            # print("text_box:",text_box)

            # torch.cuda.synchronize()
            start = time.time()

            crop_img = crop_image(org_img) # 进去 获取是否是发票的特征
            iselectric = is_electric_invoice(crop_img)
            if iselectric : # 电子发票
                print("电子发票")
                xx = EI.get_params()
                bboxes, rects, text = predict_bbox(xx, model, org_img, img, slice)
                # print("bboxes:",bboxes)
                # print("text:",text)
                # print("bboxes:",bboxes) # 4个点
                # print("rects:",rects)
                # print("text:",text)
                data = crnnRec(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB), rects)
                # print("data:",data)
                # data = [{'cx': 2425.999755859375, 'cy': 1841.4998779296875, 'text': '销o镇售方:(', 'candidate': [[], [], [], ['旧', '传', '恰', '仙', '代', '任', '信', '佳', '伟'], ['万', '分', '元', '六'], [], []], 'w': 284.0, 'h': 62.0, 'degree': -87.70103454589844}, {'cx': 1827.0, 'cy': 1832.5, 'text': '开票人:', 'candidate': [['册', '肝', '盱', '昕', '所', '脐', '牙', '升', '无'], ['飘', '察', '德', '累', '痛', '别', '荣', '制', '蔡'], ['入', '八', '从', '为', '木', '小', '不', '风', '大'], []], 'w': 181.0, 'h': 54.0, 'degree': -90.0}, {'cx': 1314.0, 'cy': 1831.5, 'text': '复核:', 'candidate': [['夏', '氨', '发', '凤', '友', '镇'], ['校', '孩', '枚', '构', '被', '棱', '筱', '枝', '接'], []], 'w': 170.0, 'h': 51.0, 'degree': -90.0}, {'cx': 2004.0, 'cy': 1824.0, 'text': '陈菁', 'candidate': [['际', '慨', '院', '你', '猴', '砀', '睐', '砾', '畅'], ['臂', '著', '膏', '背', '青', '脊', '胥', '昔', '誉']], 'w': 88.0, 'h': 44.0, 'degree': 0.0}, {'cx': 1453.5, 'cy': 1825.0, 'text': '田向君', 'candidate': [['由', '团', '阳', '四', '旧', '因', '园', '岳', '用'], ['问', '尚', '句', '古', '何', '首', '间', '同', '夜'], ['看', '碧', '老', '著', '舍', '若', '着', '和', '者']], 'w': 135.0, 'h': 48.0, 'degree': -90.0}, {'cx': 646.0, 'cy': 1824.0, 'text': '收款人:', 'candidate': [['妆', '敝', '漱', '放', '救', '激', '肽', '欣', '减'], ['献', '默', '歌', '就', '脉', '歉', '欺', '散', '欢'], ['从', '入', '八', '食', '大', '风', '为', '久', '儿'], []], 'w': 170.0, 'h': 50.0, 'degree': -90.0}, {'cx': 828.099365234375, 'cy': 1824.0172119140625, 'text': '叶关不', 'candidate': [['呼', '吁', '时', '盱', '听'], ['美', '夹', '天', '头', '吴', '兴', '买', '羌', '尖'], ['禾', '木', '本', '示', '术', '布', '衣', '千', '尔']], 'w': 152.0, 'h': 62.0, 'degree': -2.4195094108581543}, {'cx': 2417.0, 'cy': 1782.9998779296875, 'text': '发票专用章', 'candidate': [[], ['别', '惠', '无', '察', '张', '务', '聚', '制', '那'], ['吉', '支', '钟', '转', '诗', '节', '传', '寺', '式'], ['思'], ['草', '育', '票', '音', '事', '计', '童', '真', '看']], 'w': 207.0, 'h': 66.0, 'degree': -83.27310943603516}, {'cx': 1385.0001220703125, 'cy': 1768.5, 'text': '中国建设银行桐庐支行33001617135053009011', 'candidate': [['市', '由', '申', '冲', '内', '电', '单', '丰', '南'], ['园'], ['理', ',', '已'], ['识', '谡', '议', '级', '娑', '圾', '嫒', '睽', '蝾'], ['镇', '根', '锲', '录', '很', '佩', '粮', '锝', '饽'], [], ['相', '枫', '榈', '柯', '利', '柄', '枢', '板', '科'], ['户', '炉', '店', '贷', '沪', '启', '声', '房', '序'], ['专', '克', '丰', '昔', '吉', '技', '古', '曰', '式'], ['于'], ['0', '2'], [], [], [], [], [], [], [], [], [], [], ['O'], [], [], [], [], ['0'], ['O'], [], ['I', 'l', 'L']], 'w': 860.0, 'h': 50.0, 'degree': -89.8655014038086}, {'cx': 763.42626953125, 'cy': 1752.009521484375, 'text': '开户行及账号:', 'candidate': [['册', '牙'], ['产', '卢', '庐', '沪', '岸', '芦', '牌', '洋', '尸'], ['于', '管', '宁', '付', '村', '针', '产', '仔', '打'], ['乃'], ['帐', '胀', '味', '然', '胜', '躲', '联'], ['甥', '岁', '兮', '蝌', '弓', '男', '受', '毕', '蚓'], []], 'w': 282.0, 'h': 62.0, 'degree': -88.736328125}, {'cx': 1311.5, 'cy': 1713.5, 'text': '桐庐县城富春路205号0571-86833250', 'candidate': [['柯', '稍', '楠', '榈', '稿', '糊', '枸', '稠', '构'], ['卢', '炉', '泸', '贷', '沪', '嗜', '咛', '哨', '痒'], ['基', '兵', '旦', '昙', '悬', '具', '兴', '堪', '坛'], ['域', '姒', '圳', '坝', '讣', '坟', '地', '汰', '认'], ['宫', '言', '官', '宣', '句', '宜', '赏', '盲', '宴'], ['着', '眷', '卷', '举', '昔', '睿', '看', '脊', '鲁'], ['略', '咯', '晗', '眙', '贻', '酪', '骼', '赂', '赃'], [], ['O'], [], ['与', '受', '乌', '导', '岁', '以', '兮', '写'], ['O'], [], [], [], [], ['3', '2'], [], ['3'], [], [], [], [], []], 'w': 702.0, 'h': 42.0, 'degree': -90.0}, {'cx': 2250.5, 'cy': 1707.0, 'text': '', 'candidate': [], 'w': 76.0, 'h': 37.0, 'degree': -18.178016662597656}, {'cx': 788.5054931640625, 'cy': 1699.7896728515625, 'text': '址、电话:', 'candidate': [['扯', '止', '拙', '让', '她', '灿', '泄', '汕', '业'], ['。'], ['申', '鬼', '包'], ['诺', '活', '括', '浩', '诱', '洽', '洁', '培', '啥'], []], 'w': 219.0, 'h': 45.0, 'degree': -88.49256896972656}, {'cx': 650.0, 'cy': 1696.0001220703125, 'text': '地', 'candidate': [['也', '她', '他', '堆', '址', '始', '比', '悬', '灿']], 'w': 43.0, 'h': 43.0, 'degree': -14.42077350616455}, {'cx': 2436.000244140625, 'cy': 1707.0001220703125, 'text': '9130123569873055', 'candidate': [[], [], ['8', '9', '0'], ['o'], ['4', '/'], ['9'], ['5', '8', '1'], [], [], [], ['9', '6', '5', '3', '0'], [], ['8'], [], ['6'], []], 'w': 366.0, 'h': 80.0, 'degree': -81.70953369140625}, {'cx': 2245.5, 'cy': 1658.5, 'text': '', 'candidate': [], 'w': 27.0, 'h': 29.0, 'degree': -90.0}, {'cx': 1322.5, 'cy': 1656.0, 'text': '91330122589878085J', 'candidate': [['0', '8', '3', '2', '1', 'g'], ['l'], [], [], ['o'], ['l', 'I'], [], [], ['9', '6'], ['3'], ['g', '5', 'r', 's', 'i', 'h', 'o'], ['3'], [], ['3', '6', '9', 'B', '5', '0'], [], ['3', '6', '9', '0', 'B'], ['9'], []], 'w': 619.0, 'h': 40.0, 'degree': -90.0}, {'cx': 761.0082397460938, 'cy': 1642.056640625, 'text': '纳税人识别号:', 'candidate': [['约', '钠', '纺', '绒', '绵', '执', '幼', '编', '的'], ['秘', '稳', '砚', '秒', '靓', '祝', '租', '脱', '积'], ['入', '八', '下', '为', '大', '从', '贝', '火'], ['织', '误', '炽', '坦', '烬', '说', '螈', '组', '螟'], ['剔', '驯', '划', '勃', '荆', '刿', '刻', '刹', '剃'], ['岁', '乎', '弓', '甥', '青', '罗', '导', '男', '于'], []], 'w': 281.0, 'h': 63.0, 'degree': -88.91908264160156}, {'cx': 572.5, 'cy': 1672.0, 'text': '', 'candidate': [], 'w': 54.0, 'h': 191.0, 'degree': -90.0}, {'cx': 1215.5, 'cy': 1593.5, 'text': '桐庐科特华实验仪器有限公司', 'candidate': [['枸', '柯', '楠', '稿', '稍', '荷', '祠', '榆', '彻'], ['卢', '声', '冉', '泸', '肉', '芦', '伪', '沣', '估'], ['利', '积', '千', '私'], ['持', '具', '将', '精', '转', '情', '村', '惊', '并'], ['集', '生', '隼', '坐', '毕', '格', '作', '科', '乡'], ['虾', '买', '奕', '卖', '灾', '虹', '妖', '突', '吹'], ['些', '蛇', '坠', '耻', '驰', '监', '蚣', '哈', '睑'], ['似', '伐', '攸', '佟', '俊', '侈', '汶', '伙', '岐'], ['罂', '嚣', '翌', '踝', '露', '噩', '踞', '罄', '船'], ['者', '市', '省', '自', '指', '春'], [], ['司'], ['同', '合', '词', '一', '几', '命', '京']], 'w': 516.0, 'h': 42.0, 'degree': -90.0}, {'cx': 861.9998779296875, 'cy': 1592.0, 'text': '称', 'candidate': [['协', '胁', '淋', '拙', '挑', '擦', '涂', '迹', '冻']], 'w': 63.0, 'h': 40.0, 'degree': -74.74488067626953}, {'cx': 651.5, 'cy': 1599.0, 'text': '名', 'candidate': [['各', '铭', '倍', '答', '皆', '钻', '备', '铂', '若']], 'w': 32.0, 'h': 55.0, 'degree': -90.0}, {'cx': 2437.49072265625, 'cy': 1606.1146240234375, 'text': '公', 'candidate': [[]], 'w': 471.0, 'h': 218.0, 'degree': -85.28846740722656}, {'cx': 3068.5, 'cy': 1531.5, 'text': '配国用', 'candidate': [['翻', '出', '正', '社', '8', '四', '邮', '注', '即'], ['里'], ['有', '南', '市']], 'w': 80.0, 'h': 30.0, 'degree': -90.0}, {'cx': 2323.0, 'cy': 1515.0, 'text': '(小写)', 'candidate': [[], ['办', '水', '亦', '常', '事', '尔', '丁', '药', '孙'], ['骂', '驾', '抒', '泻', '丐', '焉', '弩', '霄', '摔'], []], 'w': 160.0, 'h': 50.0, 'degree': -90.0}, {'cx': 370.0, 'cy': 1510.5, 'text': '', 'candidate': [], 'w': 55.0, 'h': 20.0, 'degree': -90.0}, {'cx': 1381.0, 'cy': 1505.5001220703125, 'text': '叁万壹仟圆整', 'candidate': [['垒', '仝', '垫', '挂', '拴', '差', '栓', '鲑', '推'], ['方', '石', '月', '分', '厅', '市', '历', '入', '龙'], ['或', '卖', '变', '鼓', '妻', '壶', '妙', '吟', '尝'], ['任', '什', '伴', '仔', '件', '杆', '秤', '俨', '忏'], ['四', '团', '回', '园', '网', '图', '2', '函', '国'], ['毗', '黔', '蜂', '监', '坐', '弊', '耻', '趾', '坠']], 'w': 334.0, 'h': 52.0, 'degree': -0.3515034317970276}, {'cx': 2599.0, 'cy': 1504.0, 'text': '￥31000.00', 'candidate': [['溥', '薄', '鸪', '谡', '拦', '羊', '兰', '当'], [], [], [], [], ['O'], [','], [], []], 'w': 306.0, 'h': 60.0, 'degree': -90.0}, {'cx': 834.5000610351562, 'cy': 1504.0, 'text': '价税合计(大写)', 'candidate': [['介', '份', '巾', '优', '犹', '你', '阶', '允', '殆'], ['秘', '碰', '秒', '靓', '砚', '秕', '榄', '祝', '概'], ['含', '全', '令', '命', '拾', '舍', '仑', '给', '企'], ['斗', '升', '让', '叶', '讲', '站', '汁', '识', '讯'], [], ['达', '土', '次', '太', '头', '士', '认', '水', 'X'], ['与', '抒', '号', '耳', '雪', '泻', '骂', '霄', '宫'], []], 'w': 364.0, 'h': 60.0, 'degree': -89.02897644042969}, {'cx': 949.0, 'cy': 1425.5, 'text': '计', 'candidate': [[]], 'w': 28.0, 'h': 27.0, 'degree': -90.0}, {'cx': 2796.007568359375, 'cy': 1420.813720703125, 'text': '￥4275.8日', 'candidate': [['溥', '羊', '薄', '丫', '掺', '鸪', '*', '谡', '+'], ['A'], [], [], [], [','], [], []], 'w': 239.0, 'h': 51.0, 'degree': -1.3322197198867798}, {'cx': 2313.5, 'cy': 1422.0, 'text': '￥26724.14', 'candidate': [['羊', '当', '文', '*', '薄'], [], [], [], [], [], [','], [], ['5']], 'w': 269.0, 'h': 60.0, 'degree': -90.0}, {'cx': 3052.5, 'cy': 1257.5, 'text': '', 'candidate': [], 'w': 75.0, 'h': 187.0, 'degree': -90.0}, {'cx': 365.0, 'cy': 1133.0, 'text': '', 'candidate': [], 'w': 33.0, 'h': 79.0, 'degree': -90.0}, {'cx': 1549.0, 'cy': 1030.0, 'text': '台', 'candidate': [['合', '白', '自', '后', '治', '冶', '曰', '全', '日']], 'w': 34.0, 'h': 26.0, 'degree': -90.0}, {'cx': 2367.5, 'cy': 1028.5, 'text': '26724.14', 'candidate': [[], ['8', '0'], ['T'], [], [], [], ['L'], ['A']], 'w': 158.0, 'h': 42.0, 'degree': -90.0}, {'cx': 968.0, 'cy': 1030.0, 'text': '*试验检测机械*万能材料试验机WS-600B', 'candidate': [[], ['武', '过', '讨', '斌', '责', '成', '诚', '贰', '域'], ['些', '耻', '驻', '蛇', '吡', '驰', '趾', '监', '坠'], ['拾', '枪', '栓', '槛', '脸', '俭', '趁', '恰', '松'], ['潮', '澜', '洲', '调', '满', '濒', '涮', '浏', ']'], ['权', '枫'], ['减', '漾', '桃', '拔', '越', '沫', '襟', '伙', '微'], ['水'], ['方', '厅', '厉', '乃', '亏', '疗', '百', '行', '质'], ['电', '患', '鹿', '陆', '脆', '的', '跑', '总', '熊'], ['村', '树', '杭', '衬', '付', '朴', '针', '贵', '贯'], ['斜', '科', '制', '针', '封', '抖', '秆', '干', '籽'], ['计', '讨', '过', '武', '诚', '式', '责', '斌', '贵'], ['蛇', '哈', '蛤', '些', '耻', '吐', '驰', '坠', '驻'], ['械', '表', '枕', '枫', '杖', '根', '树', '桢', '柳'], [], [], [], ['G'], [], ['O'], ['8', 'E', '5', 'D', '6', 'R', 'b', 'S', 'H']], 'w': 758.0, 'h': 50.0, 'degree': -90.0}, {'cx': 2528.0, 'cy': 1026.0, 'text': '16%', 'candidate': [['L', 'l', '2', 'I'], ['G', '0', '5', '8'], ['<', '坻', '‰', '讪', '圻', '嗡', '乐', '地', '&']], 'w': 66.0, 'h': 36.0, 'degree': -90.0}, {'cx': 1991.5, 'cy': 1026.0, 'text': '26724.137931', 'candidate': [['3'], ['G', '8'], ['T'], [], [], [], [], [], [], ['g'], ['8'], []], 'w': 213.0, 'h': 40.0, 'degree': -90.0}, {'cx': 2844.5, 'cy': 1025.0, 'text': '4275.86', 'candidate': [['1', '2', '0', '6', '9', 'A', '5', '3'], [], ['T'], ['9', '3'], [], ['3', '2', '9', '0'], ['8', '0']], 'w': 143.0, 'h': 50.0, 'degree': -90.0}, {'cx': 2795.0, 'cy': 980.0, 'text': '额', 'candidate': [['7']], 'w': 46.0, 'h': 38.0, 'degree': -90.0}, {'cx': 2716.5, 'cy': 978.5, 'text': '税', 'candidate': [['秘', '稳', '粮', '棍', '榄', '虑', '徒', '悦', '槐']], 'w': 124.0, 'h': 42.0, 'degree': -90.0}, {'cx': 2258.0, 'cy': 975.0, 'text': '', 'candidate': [], 'w': 75.0, 'h': 21.0, 'degree': -90.0}, {'cx': 3053.856201171875, 'cy': 1054.520263671875, 'text': '', 'candidate': [], 'w': 84.0, 'h': 229.0, 'degree': -86.72950744628906}, {'cx': 2312.499755859375, 'cy': 976.0, 'text': '额', 'candidate': [['领', '颐', '颇', '颌', '欲', '新', '颔', '皈', '颖']], 'w': 51.0, 'h': 34.0, 'degree': 0.0}, {'cx': 2203.0, 'cy': 975.0, 'text': '金', 'candidate': [['全', '给', '余', '拾', '舍', '邻', '合', '仝', '主']], 'w': 26.0, 'h': 26.0, 'degree': -90.0}, {'cx': 367.5, 'cy': 993.5, 'text': 'G', 'candidate': [['C']], 'w': 32.0, 'h': 70.0, 'degree': -90.0}, {'cx': 2498.1513671875, 'cy': 975.9898071289062, 'text': '税率', 'candidate': [['秘', '稳', '棍', '虑', '祝', '杞', '患', '郴', '锐'], ['举', '卒', '挚', '奉', '鸣', '草', '吟', '拳', '癸']], 'w': 97.0, 'h': 48.0, 'degree': -3.8568007946014404}, {'cx': 1960.2890625, 'cy': 970.0145263671875, 'text': '单价', 'candidate': [['羊', '草', '曾', '弹', '岁', '的', '肖', '爱', '岸'], ['阶', '肿', '侨', '冲', '巾', '府', '你', '伤', '帕']], 'w': 128.0, 'h': 57.0, 'degree': -87.11357879638672}, {'cx': 1720.0, 'cy': 971.5, 'text': '数量', 'candidate': [[], ['一']], 'w': 121.0, 'h': 54.0, 'degree': -90.0}, {'cx': 1529.5, 'cy': 970.0, 'text': '单位', 'candidate': [['曾', '弹', '弟', '草', '销', '史', '岁', '羊', '禅'], ['俭', '住', '拉', '佐', '仕', '性', '应', '检', '睑']], 'w': 92.0, 'h': 45.0, 'degree': -90.0}, {'cx': 1299.0, 'cy': 967.5000610351562, 'text': '规格型号', 'candidate': [['观', '挪', '嘿', '巍', '抑', '鳙', '龛', '蛹', '嘌'], ['裕', '路', '牲', '释', '洛', '俗', '摇', '恪', '胳'], ['圣', '把', '垫', '珍', '濯', '碳', '瞿', '泗', '裂'], ['弓', '受', '导', '另', '岁', '月', '台', '寻', '局']], 'w': 189.0, 'h': 51.0, 'degree': -88.10139465332031}, {'cx': 844.4999389648438, 'cy': 962.4999389648438, 'text': '货物或应税劳务、服务名称', 'candidate': [['贷', '贸', '贺', '质', '负', '贵', '资', '贫', '行'], [], ['武', '哉', '威', '戍', '截', '试', '成', '咸', '忒'], ['脑', '位', '陆', '立', '庙', '成', '总', '慰', '泣'], ['租', '秘', '祝', '锐', '杞', '脱', '靓', '稳', '鹿'], ['坊', '费', '朔', '筑', '旁', '弟', '责', '肃', '芳'], ['斧', '芬', '穷', '著', '境', '坊', '劳', '芳', '夸'], [], ['股', '胆', '照', '限'], ['并', '斧', '脊', '弃', '坊', '升', '芬', '著', '劳'], ['君', '招', '多', '相', '为', '若', '书'], ['标']], 'w': 491.0, 'h': 49.0, 'degree': -89.40689086914062}, {'cx': 3077.5, 'cy': 933.0000610351562, 'text': '自到', 'candidate': [[], ['动']], 'w': 88.0, 'h': 48.0, 'degree': -82.87498474121094}, {'cx': 365.0, 'cy': 917.5, 'text': '', 'candidate': [], 'w': 51.0, 'h': 20.0, 'degree': -90.0}, {'cx': 1331.5, 'cy': 904.9999389648438, 'text': '中国农业银行连源支行610901040004708', 'candidate': [['内', '市', '小'], ['图', '里', '围', '回'], ['衣', '灾', '宋', '寂', '穴', '牧', '贫', '众', '轧'], ['北', '亚', '止', '水', '性', '金', '立', '正', '化'], ['镇', '锲', '镊', '铱', '铜', '钣', '傈', '镶', '艮'], ['石'], ['注', '涟', '洼', '旌', '溶', '谁', '淮', '洁', '涯'], ['酒', '通', '诵', '识', '漂', '啄', '涌', '诉', '泺'], ['专', '友', '号', '克', '皮', '技', '直', '乡', '料'], ['万', '于', '符', '管', '贫', '竹', '宜', '货', '气'], [], [], [], ['0'], [], [], [], [], [], [], [], [], [], [], ['3', 'B', '6', '9']], 'w': 751.0, 'h': 40.0, 'degree': -89.76927185058594}, {'cx': 1931.5001220703125, 'cy': 897.5, 'text': '主', 'candidate': [['各', '千', '乎', '全', '七', '苍', '在', '连', '轻']], 'w': 37.0, 'h': 41.0, 'degree': -45.0}, {'cx': 788.5001220703125, 'cy': 895.5001220703125, 'text': '开户行及账号:', 'candidate': [['升'], ['产', '沪', '卢', '炉', '庐', '序', '冲', '洋', '牌'], [], ['乃', '夏', '腺', '碍', '脲', '皮', '玻', '鹏', '版'], ['帐', '胀', '躲', '略', '哚', '赃', '畔', '跺', '妖'], [], []], 'w': 300.0, 'h': 54.0, 'degree': -87.24562072753906}, {'cx': 2411.5, 'cy': 870.5, 'text': '27-54<38799473>452+9>>8>907', 'candidate': [['3'], [], [], [], [], ['*', '>', '~', '+', '×', ']', '%', '沤', '_'], [], ['3', '9'], [], [], [], [], ['T'], [], ['*', '<', '/', '+'], [], [], [], ['*'], [], ['<', '*', '+', '$', '=', '_', '|', '蚪', '~'], ['<', '*', '蚪', '+'], ['3'], ['<', '*', '/', '+', 'J', '愕', ';'], [], ['O'], []], 'w': 815.0, 'h': 51.0, 'degree': -90.0}, {'cx': 1225.5, 'cy': 845.5, 'text': '涟源市桥头河镇15073867818', 'candidate': [['注', '旌', '淮', '莲', '连', '涎', '淫', '滁', '洼'], ['涿', '濠', '潦', '漂', '瀛', '喙', '凛', '泺', '啄'], [], ['析', '侨', '杆', '赉', '标', '耘', '桁', '矫', '杯'], ['兴', '类', '尖', '实', '关', '半', '夹', '斗', '学'], ['汀', '诃', '迥', '谓', '渭', '漓', '洵', '涧', '淌'], ['夫', '领', '檩', '铡', '真', '实', '椁', '颉', '钕'], [], [], ['8', 'O'], [], ['8'], ['3', '0', '9', '2'], [], [], [], [], []], 'w': 529.0, 'h': 39.0, 'degree': -90.0}, {'cx': 774.4886474609375, 'cy': 834.499755859375, 'text': '地址、电话:', 'candidate': [['也', '址', '批', '她', '兆', '他', '灿', '扯', '选'], ['扯', '批', '灿', '她', '止', '蚣', '帐', '堆', '拙'], ['。', ',', '”', '》'], ['包', '优', '龟', '仓', '韦', '申', '屯', '皂', '虑'], ['诺', '活', '括', '语', '诱', '培', '诘', '治', '诸'], []], 'w': 273.0, 'h': 54.0, 'degree': -88.69805145263672}, {'cx': 2414.0, 'cy': 823.0, 'text': '21302<5<<750*4933>2>4462></', 'candidate': [[], [], [], ['O', '9', 'o'], [], ['>', '*', '~', '+', '%', '$', '_', '×', '&'], ['9'], ['*', '>', '×', '~', '_', '[', '+', '骺', '腼'], ['*', '>', '~', '+', ';'], [], [], ['O', '6', 'G', 'o', '9'], [], [], [], [], [], ['<', '*'], [], ['<', '*', 'J', '+', '/', '_', '义', '|', ';'], [], [], [], [], ['<', '*', '+', 'J', '/', '蚪', ';', 'X', '舛'], ['>', '*', '~', '%', '洄', '=', '+', '×', '_'], ['1']], 'w': 810.0, 'h': 50.0, 'degree': -90.0}, {'cx': 1935.5, 'cy': 814.0, 'text': '码', 'candidate': [['玛', '磴', '碍', '碣', '硒', '冯', '杩', '砧', '超']], 'w': 41.0, 'h': 36.0, 'degree': -90.0}, {'cx': 364.0, 'cy': 815.5, 'text': '', 'candidate': [], 'w': 24.0, 'h': 31.0, 'degree': -90.0}, {'cx': 1311.442626953125, 'cy': 780.4985961914062, 'text': '91431382785352080K', 'candidate': [['0', 'g', '3', '2', '8'], ['l', 'n', 'I'], [], [], ['l', 'I'], [], ['3', '9'], [], [], ['9', '3', '0', '6', 'g', '5'], ['9'], ['5'], [], [], ['O', 'o', '9'], ['6', '3', '0', 'B', '9'], ['O', 'o'], ['R']], 'w': 618.0, 'h': 42.0, 'degree': -0.18754687905311584}, {'cx': 2408.94384765625, 'cy': 775.4988403320312, 'text': '1*18>9<1533*02+->>890-2+65-', 'candidate': [['I', 'l', 'L', 'T'], ['+', '<', '>'], [], ['3', 'B'], ['*', '<', '/', '+', ';', '~', '$', '=', '卅'], [], ['*', '>', '~', '%', '+', '|', '骺', '骶', '×'], [], [], [], [], ['+', '<', 'x', 'X', '>', '洲', '%'], ['o', 'O'], [], ['*', '4', '<', '>', '岫', '啪', '蚶', '姗', '咄'], [], ['<', '*', '+', '蚪', '$', '=', '|', '_', '~'], ['<', '*', '+', '蚪', '卅', '_', '》', '|', '\\'], ['3', '9'], [], ['O'], [], [], ['*', '<', '赡', '踹', '嗤', '赠', '哔', '妨', '瞄'], [], [], []], 'w': 811.0, 'h': 54.0, 'degree': -0.14288194477558136}, {'cx': 773.5, 'cy': 772.0000610351562, 'text': '纳税人识别号:', 'candidate': [['约', '钠', '执', '幼', '绒', '的', '绀', '纺', '绵'], ['秘', '祝', '积', '锐', '秒', '裘', '稳', '视', '租'], ['入', '八', '不', '从', '为', '大', '下', '个'], ['织', '误', '只', '坦', '坝', '兵', '炽', '坪', '螟'], ['荆', '剖', '剂', '刑', '剔', '犹', '剃', '涮', '刹'], ['岁', '考', '垮', '写', '另', '兮', '尊', '详', '乎'], []], 'w': 275.0, 'h': 54.0, 'degree': -88.92314147949219}, {'cx': 580.5, 'cy': 806.0, 'text': '', 'candidate': [], 'w': 54.0, 'h': 181.0, 'degree': -90.0}, {'cx': 2412.5, 'cy': 728.5, 'text': '+457+2*52--<372+/642>9/05-9', 'candidate': [['*', '￥', '~', '<', '嫡', '捌', '媾', '>', '〉'], [], [], [], ['*', '<', '踮', '嗤', '瞄', '哒', '咄', '哔', '啪'], [], ['x', '水'], [], [], [], [], ['*', '>', '~', '+', '[', '_', 'Z', '骶', '忑'], [], ['T'], [], ['*'], [], [], [], [], [], [], ['1'], ['O'], [], [], []], 'w': 817.0, 'h': 51.0, 'degree': -90.0}, {'cx': 1258.0, 'cy': 722.5, 'text': '注源市天龙电力电杆制造有限公司', 'candidate': [['涟', '淮', '浩', '涎', '涯', '旌', '谁', '澄', '淫'], ['诵', '漂', '谓', '酒', '潦', '泺', '涿', '诋', '蚤'], [], ['关', '无', '夫', '夭', '禾', '秃', '深', '夹', '采'], [], ['比', '虫', '申', '韦', '包', '曳', '曲', '屯', '龟'], ['方', '加', '九', '为', '首', '大', '刀', '六', '苏'], ['申', '曳', '包', '盅', '皂', '鬼', '屯', '虫', '饱'], ['秆', '秤', '粹', '砰', '矸', '俨', '肝', '怦', '柘'], ['刷', '荆', '剃', '刺', '削', '剥', '剐', '剌', '划'], ['谐', '逍', '遗', '遣', '逾', '汕', '渔', '遁', '谊'], ['者', '不', '布', '省', '市', '角', '看', '若', '何'], [], [], ['同', '词', '局']], 'w': 595.0, 'h': 42.0, 'degree': -90.0}, {'cx': 878.0, 'cy': 711.0, 'text': '称:', 'candidate': [['栋', '愁', '秋', '怅', '态', '脉', '标', '桃', '意'], []], 'w': 65.0, 'h': 45.0, 'degree': -90.0}, {'cx': 2392.467529296875, 'cy': 624.0115356445312, 'text': '开票日期:', 'candidate': [['册', '肝', '昕', '脐', '所', '牙', '盱', '无', '掰'], ['飘', '制', '氯', '累', '察', '农', '熏', '剩', '痛'], ['曰', '口', '白'], ['斯', '阳', '行', '册', '开', '乎', '嘴', '朗', '职'], []], 'w': 210.0, 'h': 51.0, 'degree': -88.58206939697266}, {'cx': 2711.2802734375, 'cy': 613.997314453125, 'text': '2018年06月15日', 'candidate': [[], ['O', 'o', '6', '8', '9', 'G'], [], [], ['车', '华', '午', '库', '作', '早', '名', '军', '炸'], ['O', 'o', '6'], [], ['万', '村', '片', '为', '信', '其', '号', '方', '后'], [], [], ['旧', '8', '白', '年', '已', '四', '田', '曰', '回']], 'w': 316.0, 'h': 54.0, 'degree': -0.5598757863044739}, {'cx': 2794.499755859375, 'cy': 547.4999389648438, 'text': '20050538', 'candidate': [[], ['9'], [], [], ['O'], [], [], ['3']], 'w': 214.0, 'h': 40.0, 'degree': 0.0}, {'cx': 2809.500244140625, 'cy': 493.0, 'text': '3300174130', 'candidate': [['2', '8'], [], ['O'], ['9'], [], [], [], [], [], ['O']], 'w': 201.0, 'h': 42.0, 'degree': -1.4763686656951904}, {'cx': 2448.5, 'cy': 488.0, 'text': 'No20050538', 'candidate': [['n', '1', 'A', 'I', 'l', 'M', 'W'], ['a', '0', 'g', 'e', 'O'], ['3', '0'], ['p'], [], ['9', 'S', 's', '3', '8', '6'], ['6', '9', '8', 'O'], ['9', 'S', 's', '3', '1', '8', '6'], ['9', '2', '8', '5', '7', '0', '1'], ['0', '6', '9', '3', 'B', '5', 'E', '日', 'S']], 'w': 484.0, 'h': 70.0, 'degree': -89.27324676513672}, {'cx': 1018.5, 'cy': 482.0, 'text': '3300174130', 'candidate': [[], ['8', '2'], [], ['O', 'o'], [], [], [], [], [], ['O', 'o', '9', '6']], 'w': 413.0, 'h': 62.0, 'degree': -90.0}, {'cx': 1880.5, 'cy': 442.0, 'text': '+', 'candidate': [['士', '主', '土', '十', '垛', '诠', '诤', '岫', '绦']], 'w': 30.0, 'h': 33.0, 'degree': -90.0}, {'cx': 1970.5, 'cy': 447.0, 'text': '用发票', 'candidate': [['月', '君', '员', '相', '开', '和', '司', '看', '者'], [], ['制', '飘', '熟', '察', '部', '宋', '耐', '颜', '朝']], 'w': 214.0, 'h': 97.0, 'degree': -90.0}, {'cx': 1493.4998779296875, 'cy': 441.0, 'text': '浙江增', 'candidate': [['淅', '渐', '沂', '断', '晰', '逝', '沥', '嘶', '浒'], ['工', '汇', '汀', '汪'], []], 'w': 249.0, 'h': 82.0, 'degree': 0.0}]
                # print("data:",data)
                # for i in range(0,len(data)):
                #     print(data[i]['cx'],data[i]['cy'])
                result = EI.formatResult(data)
                print("result:",result)
            else :
                # print("非电子发票")
                fg_img, bg_img = split_fgbg_from_image(crop_img)
                text_box = crop_img.copy()

                bg_scaled = data_loader.scale(bg_img)
                bg_bboxes, bg_rects, text = predict_bbox(args, model, crop_img, bg_scaled, slice)
                bg_data = crnnRec(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB), bg_rects)

                fg_scaled = data_loader.scale(fg_img)
                fg_bboxes, fg_rects, text = predict_bbox(args, model, crop_img, fg_scaled, slice)
                fg_data = crnnRec(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), fg_rects)

                fg_data, fg_bboxes = layout_adjustment(fg_data, fg_bboxes, crop_img, args, model, slice,
                                                       data_loader.scale)

                bboxes = fg_bboxes
                result = formatResult(bg_data, fg_data)

            # 以下是保存和准确率的计算
            # torch.cuda.synchronize()
            end = time.time()
            total_frame += 1
            total_time += (end - start)
            print('fps: %.2f' % (total_frame / total_time))
            sys.stdout.flush()

            for bbox in bboxes :
                cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
            print("data_loader.img_paths:",data_loader.img_paths)
            image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
            write_result_as_txt(image_name, bboxes, 'outputs/submit_invoice/')

            text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
            debug(idx, data_loader.img_paths, [[text_box]], 'data/images/tmp/')

            # result = crnnRec(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB), rects)
            # result = formatResult(bg_data, fg_data)

            if args.evaluate :
                image_file = image_name + '.txt'
                error_file = image_name + '-errors.txt'
                file = os.path.join(os.path.dirname(__file__), 'data/gt/', image_file)
                # print("file:",file)
                errfile = os.path.join(os.path.dirname(__file__), 'data/error/', error_file)
                # print("errfile:",errfile)
                if os.path.exists(file) :
                    precision = evaluate(file, errfile, result)
                    print('precision1:' + str(precision[0]) + '%')
                    precisions1.append(precision[0])

                if not iselectric :
                    bg_file = image_name + '_background.jpg'
                    file = os.path.join(os.path.dirname(__file__), 'data/error/images/', bg_file)
                    cv2.imwrite(file, bg_img)
                    fg_file = image_name + '_foreground.jpg'
                    file = os.path.join(os.path.dirname(__file__), 'data/error/images/', fg_file)
                    cv2.imwrite(file, fg_img)
        except Exception as e :
            print(e)
    # cmd = 'cd %s;zip -j %s %s/*' % ('./outputs/', 'submit_invoice.zip', 'submit_invoice')
    # print(cmd)
    sys.stdout.flush()
    # util.cmd.Cmd(cmd)
    if len(precisions1) :
        print("成功识别发票的个数:",len(precisions1))
        mean = np.mean(precisions1)
        print('mean precision1:' + str(mean) + '%')

    return result


def predict_bbox(args, model, org_img, img, slice) :
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

    if args.arch == 'mobilenet' :
        pred = pse2(kernels, args.min_kernel_area / (args.scale * args.scale))
    else :
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
    for i in range(1, label_num) :
        points = np.array(np.where(label == i)).transpose((1, 0))[:, : :-1]

        # print("points.shape:",np.shape(points))
        # print("points:",points)
        if points.shape[0] < args.min_area / (args.scale * args.scale) :
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score :
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


def is_electric_invoice(crop_image) :
    h, w, c = crop_image.shape
    l = int(w * 0.08)
    r = int(w * 0.125)
    t = int(h * 0.02)
    b = int(h * 0.125)
    img = crop_image[t :b, l :r, :]

    show_image('bar code', img)

    color_dict = {
        'blue' : [np.array([100, 43, 46]), np.array([125, 255, 255])],
        'black' : [np.array([0, 0, 0]), np.array([180, 255, 46])],
        # 'white': [np.array([0,0,221]), np.array([180,30,255])],
    }

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maxsum = 500
    color = None

    for d in color_dict :
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations = 2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts :
            sum += cv2.contourArea(c)
        if sum > maxsum :
            maxsum = sum
            color = d

    return color == 'black'


def show_image(winname, img) :
    # cv2.namedWindow(winname, 0)
    # cv2.resizeWindow(winname, 800, 600)
    # cv2.imshow(winname, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('data/images/tmp/'+winname+'.jpg', img)
    pass


def calc_std(data) :
    a = np.array(data)
    b = np.mean(a)
    c = a - b
    d = c ** 2
    e = np.mean(d)
    f = e ** 0.5

    return f


def gamma_trans(img) :  # gamma函数处理
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(grey)
    gamma = math.log10(mean / 255) / math.log10(0.5)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def get_dark_area(img, bright) :
    contours, hierarchy = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    contour = contours[0]

    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [approx], -1, 255, -1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    target = cv2.bitwise_and(img, img, mask = mask)
    target = gamma_trans(target)

    return target


def get_color_mask(img) :
    color_dict = {
        'blue' : [np.array([100, 43, 46]), np.array([125, 255, 255])],
        # 'blue' : [np.array([69, 43, 46]), np.array([125, 255, 255])],
        'black' : [np.array([0, 0, 0]), np.array([180, 255, 46])],
        'grey' : [np.array([0, 0, 46]), np.array([180, 10, 110])]
    }

    h, w, c = img.shape
    boxes = [
        [h // 2 - h // 8, h // 2 + h // 8, 0, w // 4],  # left
        [h // 2 - h // 8, h // 2 + h // 8, w * 3 // 4, w],  # right
    ]
    # t = h // 2 - h // 8
    # b = h // 2 + h // 8
    # l = w // 4 - w // 8
    # r = w // 4 + w // 8
    # crop = img[t:b, l:r, :]
    # hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # main_color = [255, 255, 255]
    text_color = 'blue'

    # for box in boxes:
    #     t,b,l,r = box
    #     crop = img[t:b, l:r, :]
    #
    #     # image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    #     # small_image = image.resize((80, 80))
    #     # result = small_image.convert('P', palette=Image.ADAPTIVE, colors=5)  # image with 5 dominating colors
    #     # result = result.convert('RGB')
    #     # # result.show() # 显示图像
    #     # main_colors = result.getcolors(80 * 80)
    #     # main_colors = sorted(main_colors, key=lambda x: x[0])
    #     # main_color = min([main_color, list(main_colors[0][1])])
    #
    #     if text_color != 'blue':
    #         hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    #         maxsum = 0
    #         for d in color_dict:
    #             mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
    #             binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    #             binary = cv2.dilate(binary, None, iterations=2)
    #             cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             sum = 0
    #             for c in cnts:
    #                 sum += cv2.contourArea(c)
    #             if sum > maxsum:
    #                 maxsum = sum
    #                 text_color = d

    # print('main color: {}'.format(main_color))
    print('text color: {}'.format(text_color))

    if text_color == 'blue' :
        normalized, scale = trans_image(img)
        # yuv = cv2.cvtColor(normalized, cv2.COLOR_BGR2YUV)
        # Y, U, V = cv2.split(yuv)
        #
        # mean = round(np.mean(Y))
        # std = calc_std(Y)

        # print('mean Y: {}, stand deviance Y: {}'.format(mean, std))

        B, G, R = cv2.split(normalized)
        mean = round(np.mean(B))
        std = calc_std(B)

        print('mean B: {}, stand deviance B: {}'.format(mean, std))

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(yuv)
        if mean > 1 or 0.35 < std < 0.41 :
            mask = cv2.inRange(U, 135, 255)
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(hsv, color_dict[text_color][0], color_dict[text_color][1])
            # grey_mask = cv2.inRange(hsv, color_dict['black'][0], color_dict['black'][1])
            # mask = cv2.bitwise_or(mask, grey_mask)
        else :
            mask = cv2.inRange(U, 130, 255)
            bright = cv2.inRange(Y, 150, 255)
            dark = get_dark_area(img, bright)
            hsv = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
            bmask = cv2.inRange(hsv, np.array([90, 0, 43]), np.array([150, 255, 255]))
            mask = cv2.bitwise_or(mask, bmask)

    elif text_color == 'black' :
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_dict[text_color][0], color_dict[text_color][1])
        grey_mask = cv2.inRange(hsv, color_dict['grey'][0], color_dict['grey'][1])
        mask = cv2.bitwise_or(mask, grey_mask)

    return mask


def crop_image(org_img) :
    h, w, c = org_img.shape

    points = vggloc.get_points(org_img)
    t = min(points, key = lambda x : x[1])[1]
    b = max(points, key = lambda x : x[1])[1]
    l = min(points, key = lambda x : x[0])[0]
    r = max(points, key = lambda x : x[0])[0]

    dy = b - t
    dx = r - l
    t = max([0, int(t - dy * 0.3)])
    b = min([h, int(b + dy * 0.14)])
    l = max([0, int(l - dx * 0.05)])
    r = min([w, int(r + dx * 0.05)])

    crop = org_img[t :b, l :r, :]
    show_image('croped', crop)
    return crop


# def get_roi(img, points):
#     t = min(points, key=lambda x: x[1])[1]
#     b = max(points, key=lambda x: x[1])[1]
#     l = min(points, key=lambda x: x[0])[0]
#     r = max(points, key=lambda x: x[0])[0]
#
#     return img[t:b, l:r, :]

def split_fgbg_from_image(img) :
    # set blue thresh 设置HSV中蓝色、天蓝色范围
    mask_color = [255, 255, 255]

    mask = get_color_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))

    dilated = cv2.dilate(mask, kernel)
    bg_img = np.copy(img)
    bg_img[dilated != 0] = mask_color

    fg_img = np.copy(img)
    fg_img[dilated == 0] = mask_color

    show_image('background', bg_img)
    show_image('foreground', fg_img)

    return fg_img, bg_img


def layout_adjustment(target, tboxes, img, args, model, slice, scaleFunc) :
    layouts = detect_layout(vggloc, img)
    if layouts['content'] is not None :
        content = layouts['content']
        replace = {}
        insert = {}

        for c in content :
            roi = get_roi(img, c)
            show_image('roi', roi)
            scaled = scaleFunc(roi)
            bboxes, rects, text = predict_bbox(args, model, roi, scaled, slice)
            # print("")
            # print("text:",text)
            data = crnnRec(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), rects)

            s = [i for i, d in enumerate(data) if re.search(r'服?务?名称$|规?格?型号$', d['text'])]
            e = [i for i, d in enumerate(data) if re.search(r'(^合$|^计$|^合计$)', d['text'])]
            s = max(s) + 1 if len(s) else 0
            e = min(e) if len(e) else len(data)

            if not s :
                break

            data = data[s :e]
            for d in data :
                t = [i for i, t in enumerate(target) if calc_axis_iou(d, t, 1) > 0.6 and (calc_axis_iou(d, t) > 0.6 or (
                            d['cx'] - d['w'] / 2 > t['cx'] - t['w'] / 2 and d['cx'] + d['w'] / 2 < t['cx'] + t[
                        'w'] / 2))]
                if len(t) :
                    k = t[0]
                    if not k in replace :
                        replace[k] = d
                    else :
                        if not k in insert :
                            insert[k] = []
                        insert[k].append(d)
                else :
                    for k, v in replace.items() :
                        y_iou = calc_axis_iou(v, d, 1)
                        if y_iou > 0.4 :
                            if not k in insert :
                                insert[k] = []
                            insert[k].append(d)
                            break

        for k, v in replace.items() :
            target[k] = v

        keys = sorted([k for k in insert], key = lambda x : x, reverse = True)
        for k in keys :
            target = target[:k + 1] + insert[k] + target[k + 1 :]

        return target, tboxes


import re


def isTitle(text) :
    return re.search(r'(货物.?|.?服务名｜服.?名称|.?务名称|规.?型号|^单.?|^数|^金.?|.?额$|^税.{0,2}$|项目|类型|车牌|通行日期)', text) != None
    # return re.search(r'(货物.?|.?服务名｜服.?名称|.?务名称|规.?型号|^单.?|^数|.?额$|^税.$|项目|类型|车牌|通行日期)', text) != None
    # return text.find(r'服务名称') > -1 \
    #        or text.find(r'规格型号') > -1 \
    #        or text.find(r'单位') > -1 \
    #        or text.find(r'数量') > -1 \
    #        or text.find(r'单价') > -1 \
    #        or text.find(r'金额') > -1 \
    #        or text.find(r'税率') > -1 \
    #        or text.find(r'税额') > -1


def isSummary(text) :
    return text == r'合' or text == r'计' or text == r'合计'


def get_content_boundary(bg_data, fg_data) :
    titles = [d for d in bg_data if isTitle(d['text'])]
    titles = [d for d in bg_data if calc_axis_iou(d, titles, 1) > 0.2]
    titles = sorted(titles, key = lambda x : x['cy'] + x['h'] / 2, reverse = True)
    if len(titles) > 8 :
        titles = titles[:8]
    # right = max(titles, key=lambda x:x['cx']+x['w']/2)
    bottom = titles[0]

    summary = [d for d in fg_data if re.search(r'(￥|Y|羊)\d+', d['text'])]
    if len(summary) > 3 :
        summary = summary[-3 :]

    right = max(summary, key = lambda x : x['cx'] + x['w'] / 2)
    summary = min(summary, key = lambda x : x['cy'] - x['h'] / 2)

    content = [d for d in fg_data if d['cy'] > (bottom['cy'] - bottom['h'] / 2) and d['cy'] < summary['cy'] and (
                d['cx'] < right['cx'] or calc_axis_iou(d, right) > 0.1)]

    while len(content) :
        left = min(content, key = lambda x : x['cx'] - x['w'] / 2)
        if len(left['text']) < 3 :
            content.remove(left)
        else :
            break

    return content


def check(data, placeholder, dir = 0) :
    try :
        i = None
        if isinstance(placeholder, list) :
            for pld in placeholder :
                f = [x for x, d in enumerate(data) if d == pld]
                if len(f) :
                    i = f[-1]
                    break
        else :
            f = [x for x, d in enumerate(data) if d == placeholder]
            if len(f) :
                i = f[-1]
        return i
    except :
        return None


LEFT_MARGIN = 70
RIGHT_MARGIN = 30


def parseLine(line, boundary) :
    xmin, xmax = boundary

    copyed = copy.deepcopy(line)
    copyed = preprocess_line(copyed)
    # ratio
    # ratio, price = get_ratio(copyed, xmax)
    ratio, price, tax = get_ratio_price_tax(copyed, xmax)
    # title
    title = get_title(copyed, xmin)
    # tax
    # tax = get_tax(copyed, xmax, ratio)
    # prices
    specs, amount, uprice, price, ratio = get_numbers(copyed, price, ratio)
    # specs
    specs, unit = get_specs_unit(copyed, specs)

    return postprocess_line(title, specs, unit, amount, uprice, price, ratio, tax)


def is_merge_str(x1, x2) :
    ret = False

    eng_pattern = re.compile(r'[a-zA-Z\s]')
    chn_pattern = re.compile(r'[\u4e00-\u9fa5]')
    num_pattern = re.compile(r'[\d\.]')

    s1 = x1['text'][-1]
    s2 = x2['text'][0]

    if num_pattern.match(s1) and num_pattern.match(s2) :
        ret = True
    elif chn_pattern.match(s1) and chn_pattern.match(s2) :
        ret = True
    elif eng_pattern.match(s1) and eng_pattern.match(s2) :
        ret = True

    return ret


def is_disturb(i, line) :
    ret = False
    pi = i - 1
    ni = i + 1

    eng_pattern = re.compile(r'[a-zA-Z\s]')
    chn_pattern = re.compile(r'[\u4e00-\u9fa5]')
    num_pattern = re.compile(r'[\d\.]')

    if ni < len(line) :
        cur = line[i]['text']
        prev = line[pi]['text']
        next = line[ni]['text']

        for pattern in [num_pattern, chn_pattern, eng_pattern] :
            if pattern.match(prev[-1]) and pattern.match(next[0]) and len(cur) <= 2 \
                    and (not pattern.match(cur[0]) or not pattern.match(cur[-1])) :
                ret = True
                break
    return ret


def correct_number(text) :
    m = re.match(r'\d{3,}\.?\d*([^\d\.])\d*\.?\d*$', text)
    if m :
        nodigit = m.group(1)
        if nodigit == '年' :
            replace = '4'
        elif nodigit == '日' :
            replace = '6'
        elif nodigit == 'F' :
            replace = '1'
        else :
            replace = ''
        text = re.sub(nodigit, replace, text)
    return text


def preprocess_line(line) :
    line = sorted(line, key = lambda x : x['cx'] - x['w'] / 2)
    res = []

    i = 0
    j = 1
    while i < len(line) and j < len(line) :
        x = line[i]
        y = line[j]
        x1 = x['cx'] + x['w'] / 2
        y1 = y['cx'] - y['w'] / 2

        if abs(x1 - y1) < 8 and i :
            if not is_disturb(j, line) :
                w = y['cx'] + y['w'] / 2 - x['cx'] + x['w'] / 2
                cx = x['cx'] - x['w'] / 2 + w / 2
                x['w'] = w
                x['cx'] = cx
                x['text'] = x['text'] + y['text']
            j = j + 1
        else :
            x1 = x['cx'] - x['w'] / 2
            x2 = x['cx'] + x['w'] / 2
            y1 = y['cx'] - y['w'] / 2
            y2 = y['cx'] + y['w'] / 2

            if x1 < y1 and x2 > y2 :
                j = j + 1
            elif x1 > y1 and x2 < y2 :
                i = j
                j = j + 1
            else :
                res.append(x)
                i = j
                j = j + 1

    res.append(line[i])

    for i in range(max(len(res) - 3, 0), len(res)) :
        res[i]['text'] = correct_number(res[i]['text'])

    return res


def postprocess_line(title, specs, unit, amount, uprice, price, ratio, tax) :
    if not tax and price and ratio :
        tax = str(round(float(price) / 100 * float(ratio[:-1]), 2))
    elif tax and price :
        r = str(int(round(float(tax) / float(price) * 100)))
        if not ratio or ratio[:-1] == '1' or (r != ratio[:-1] and r.find(ratio[:-1]) >= 0) :
            ratio = r + '%'

    return [title, specs, unit, amount, uprice, price, ratio, tax]


def get_title(line, xmin) :
    title = ''
    candidates = [d for d in line if abs(d['cx'] - d['w'] / 2 - xmin) < LEFT_MARGIN]
    if len(candidates) :
        candidates = sorted(candidates, key = lambda x : x['cy'])
        for c in candidates :
            title += c['text']
            line.remove(c)

    if title and not title.startswith('*') :
        title = '*' + title

    return title


def get_tax(line, xmax, ratio) :
    tax = ''

    if ratio == '免税' :
        tax = '***'

    idx = [i for i, d in enumerate(line) if abs(d['cx'] + d['w'] / 2 - xmax) < RIGHT_MARGIN]
    if len(idx) :
        idx = idx[0]
        ln = line[idx]
        tax = ln['text']
        line.pop(idx)

        if len(tax) > 2 and tax.find('.') == -1 :
            tax = tax[:-2] + '.' + tax[-2 :]

    return tax


def get_ratio(line, xmax) :
    ratio = ''
    price = ''
    pat = re.compile(r'(\-?[\dBG]+\.?[\dBG]{2})*(([\dBG]+[\%])|(免税))$')

    for i in range(len(line) - 1, -1, -1) :
        text = line[i]['text']
        m = pat.match(text)
        if m :
            price, ratio = (m.group(1), m.group(2)) if m.group(1) else ('', m.group(2))

        if ratio :
            line.pop(i)
            break

    if not ratio :
        numbers = sorted([i for i, d in enumerate(line) if re.match(r'([\dBG]+\.?[\dBG]{2})*', d['text'])],
                         key = lambda x : line[x]['cx'] - line[x]['w'] / 2)
        if len(numbers) >= 3 :
            i = numbers[-2]
            d = line[i]
            m = re.match(r'(\d{1,2})\D+', d['text'])
            if m :
                ratio = m.group(1) + '%'
                line.pop(i)

    return ratio, price


def get_ratio_price_tax(line, xmax) :
    ratio = ''
    price = ''
    tax = ''

    pat = re.compile(r'(\-?[\dBG]+\.?[\dBG]{2})*(([\dBG]+[\%m\u4e00-\u9fa5])|([\u4e00-\u9fa5]+税))$')
    pat2 = re.compile(r'(\-?[\dBG]+\.[\dBG]{2})([\dBG]{1,2}[8])')

    ratioitem = None
    for i in range(len(line) - 1, -1, -1) :
        text = line[i]['text']
        m = pat.match(text)
        if m :
            price, ratio = (m.group(1), m.group(2)) if m.group(1) else ('', m.group(2))
            ratio = ratio[:-1] + '%'
        else :
            m = pat2.match(text)
            if m :
                price, ratio = (m.group(1), m.group(2))
                ratio = ratio[:-1] + '%'

        if ratio :
            ratioitem = line.pop(i)
            break

    if not ratio :
        numbers = sorted([i for i, d in enumerate(line) if re.match(r'([\dBG]+\.?[\dBG]{2})*', d['text'])],
                         key = lambda x : line[x]['cx'] - line[x]['w'] / 2)
        if len(numbers) >= 3 :
            i = numbers[-2]
            d = line[i]
            m = re.match(r'(\d{1,2})\D+', d['text'])
            if m :
                ratio = m.group(1)
                ratioitem = line.pop(i)

    if re.search(r'税$', ratio) :
        tax = '***'
    else :
        if ratioitem :
            taxes = [l for l in line if l['cx'] > ratioitem['cx']]
            if len(taxes) :
                tax = taxes[0]['text']
                line.remove(taxes[0])

        if not tax :
            idx = [i for i, d in enumerate(line) if abs(d['cx'] + d['w'] / 2 - xmax) < RIGHT_MARGIN]
            if len(idx) :
                idx = idx[0]
                ln = line[idx]
                tax = ln['text']
                line.pop(idx)

                if len(tax) > 2 and tax.find('.') == -1 :
                    tax = tax[:-2] + '.' + tax[-2 :]

        if len(price) and not '.' in price :
            if tax and ratio :
                while float(price) > float(tax) :
                    prc = price[:-2] + '.' + price[-2 :]
                    f_tax = float(tax)
                    f_ratio = float(ratio[:-1])
                    f_price = float(prc)

                    if abs(f_price * f_ratio / 100.0 - f_tax) > 0.1 and f_ratio < 10 :
                        ratio = price[-1] + ratio
                        price = price[:-1]
                    else :
                        price = prc
                        break
            else :
                price = price[:-2] + '.' + price[-2 :]

    if price and ratio and not tax :
        tax = str(round(float(price) * float(ratio[:-1]) / 100.0, 2))

    return ratio, price, tax


def get_numbers(line, price, ratio) :
    specs = ''
    amount = ''
    uprice = ''

    pattern = re.compile(r'\-?[\dBG:]+\.?[\dBG]*$')
    numbers = []

    for d in line :
        if pattern.match(d['text']) :
            d['text'] = d['text'].replace('B', '8').replace('G', '6').replace(':', '')
            numbers.append(d)

    if len(numbers) :
        for n in numbers :
            line.remove(n)

        # preprocess_number(numbers)
        numbers = sorted(numbers, key = lambda x : x['cx'] - x['w'] / 2)
        if not ratio and re.match(r'^\d{2,3}$', numbers[-1]['text']) :
            ratio = numbers[-1]['text']
            ratio = ratio[:-1] + '%'
            numbers = numbers[0 :-1]
        if not price :
            price = numbers[-1]['text']
            m = re.match(r'(\d+\.\d{2})\d*(\d{2})$', price)
            if m and not ratio :
                price = m.group(1)
                ratio = m.group(2) + '%'

            if not '.' in price :
                price = price[0 :-2] + '.' + price[-2 :]

            numbers = numbers[0 :-1]

        numlen = len(numbers)
        if numlen == 3 :
            specs = numbers[0]['text']
            amount = numbers[1]['text']
            uprice = numbers[2]['text']
        elif numlen == 2 :
            num1 = numbers[0]['text']
            num2 = numbers[1]['text']

            if abs(float(num1) * float(num2) - float(price)) < 0.01 :
                specs = ''
                amount = num1
                uprice = num2
            elif abs(float(num2) - float(price)) <= 0.1 :
                specs = num1
                amount = '1'
                uprice = num2
            else :
                specs, amount, uprice = get_amount_uprice(price, num2, num1)

        elif numlen == 1 :
            specs = ''
            num = numbers[0]['text']
            if abs(float(num) - float(price)) < 0.01 :
                amount = '1'
                uprice = num
            else :
                specs, amount, uprice = get_amount_uprice(price, num)

            if not amount :
                amount = num

    return specs, amount, uprice, price, ratio


def preprocess_number(numbers) :
    number = [i for i, n in enumerate(numbers) if n['text'].find(':') > -1]
    adds = []
    removes = []
    for i in number :
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

    for d in removes :
        numbers.remove(d)
    numbers.extend(adds)


def get_min_distance(word1, word2) :
    m, n = len(word1), len(word2)
    if m == 0 : return n
    if n == 0 : return m
    cur = [0] * (m + 1)  # 初始化cur和边界
    for i in range(1, m + 1) : cur[i] = i

    for j in range(1, n + 1) :  # 计算cur
        pre, cur[0] = cur[0], j  # 初始化当前列的第一个值
        for i in range(1, m + 1) :
            temp = cur[i]  # 取出当前方格的左边的值
            if word1[i - 1] == word2[j - 1] :
                cur[i] = pre
            else :
                cur[i] = min(pre + 1, cur[i] + 1, cur[i - 1] + 1)
            pre = temp
    return cur[m]


def get_max_commonsubstr(s1, s2) :
    # 求两个字符串的最长公共子串
    # 思想：建立一个二维数组，保存连续位相同与否的状态

    len_s1 = len(s1)
    len_s2 = len(s2)

    # 生成0矩阵，为方便后续计算，多加了1行1列
    # 行: (len_s1+1)
    # 列: (len_s2+1)
    record = [[0 for i in range(len_s2 + 1)] for j in range(len_s1 + 1)]

    maxNum = 0  # 最长匹配长度
    p = 0  # 字符串匹配的终止下标

    for i in range(len_s1) :
        for j in range(len_s2) :
            if s1[i] == s2[j] :
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1

                if record[i + 1][j + 1] > maxNum :
                    maxNum = record[i + 1][j + 1]
                    p = i  # 匹配到下标i

    # 返回 子串长度，子串
    return maxNum, s1[p + 1 - maxNum : p + 1]


def get_amount_uprice(price, upricecand, amtcand = None) :
    price = float(price)
    specs = ''
    amount = ''
    uprice = ''

    copyprice = upricecand
    dotplace = upricecand.find('.')
    if dotplace > 0 :
        upricecand = upricecand[:dotplace] + upricecand[dotplace + 1 :]

    if amtcand :
        upr = str(math.trunc(float(price) / float(amtcand) * 100))
        idx = upricecand.find(upr)
        if idx >= 0 :
            amount = amtcand
            upricecand = upricecand[idx :]
            dot = len(upr[:-2])
            uprice = upricecand[:dot] + '.' + upricecand[dot :]

    if not uprice :
        end = dotplace - 1 if dotplace > 0 else len(upricecand) - 2
        for idx in range(2, end) :
            amt = int(upricecand[:idx])
            upr = upricecand[idx :]
            if not amt :
                break

            calcupr = price / amt
            if calcupr < 1 :
                break
            dot = str(calcupr).find('.')
            if dot > len(upr) :
                break
            upr = float(upr[0 :dot] + '.' + upr[dot :])
            dis = abs(upr - calcupr)
            if dis < 1 :
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

        if not uprice :
            m = re.match(r'(\d+0+)([1-9]\d*\.\d+)', copyprice)
            if m :
                amount = m.group(1)
                uprice = m.group(2)
            else :
                m = re.search(r'(\d+)(0{2,})(\d+)', upricecand)
                if m :
                    amount = m[1] + m[2]
                    uprice = m[3]
                    calcupr = str(price / float(amount))
                    n, s = get_max_commonsubstr(calcupr, uprice)
                    uprice = calcupr[:calcupr.find(s)] + uprice[uprice.find(s) :]
                else :
                    uprice = copyprice
                    if amtcand :
                        amount = amtcand
                    else :
                        amount = str(int(price / float(uprice)))
        else :
            if amtcand :
                specs = amtcand

    return specs, amount, uprice


def get_specs_unit(line, specs) :
    unit = ''
    linelen = len(line)
    if linelen >= 2 :
        specs = line[0]['text']
        unit = line[1]['text']
    if linelen == 1 :
        text = line[0]['text']
        if specs :
            unit = text
        else :
            if len(text) == 1 :
                unit = text
            else :
                specs = text

    return specs, unit


def zh_count(str) :
    count = 0
    for s in str :
        if '\u4e00' <= s <= '\u9fff' :
            count += 1
    return count


def is_wrapped_title(data, line, content, next, boundary) :
    ret = False
    xmin = boundary[0]
    dx = abs(data['cx'] - data['w'] / 2 - xmin)
    text = data['text']

    if dx < LEFT_MARGIN :
        if len(text) < 6 and len(line) :
            ret = True
        else :
            if not re.search(r'\*[\u4e00-\u9fa5]', text) :
                if next == len(content) :
                    if (not len(line) or calc_axis_iou(data, line) > 0.2) :
                        ret = True
                else :
                    end = min([next + 8, len(content)])
                    neighbor = None
                    title = None
                    for c in content[next :end] :
                        if calc_axis_iou(c, data) > 0.2 :
                            title = c
                        else :
                            if neighbor is None :
                                neighbor = c
                            else :
                                dx = c['cx'] - data['cx']
                                if (neighbor['cx'] - data['cx'] > dx > 0) and calc_axis_iou(c, neighbor) < 0.1 :
                                    neighbor = c

                    if neighbor is None :
                        ret = True
                    else :
                        y_iou = calc_axis_iou(neighbor, data, 1)
                        if y_iou > 0.3 :
                            if calc_axis_iou(neighbor, line, 1) > 0.1 and calc_axis_iou(neighbor,line) < 0.1 and calc_axis_iou(data, line) > 0.1 :
                                ret = True
                        elif y_iou < 0.1 :
                            ret = True
                        else :
                            if title is not None :
                                if calc_axis_iou(neighbor, title, 1) > 0.25 :
                                    ret = True
                            else :
                                if calc_axis_iou(data, line) > 0.2 :
                                    ret = True

    # if dx < LEFT_MARGIN and len(line) and calc_axis_iou(data, line) > 0.2 and (len(text) < 5 or not re.search(r'\*[\u4e00-\u9fa5]', text)):
    #     ret = True

    return ret


def check_title(elem, line, data, start, end, boundary) :
    ret = False
    xmin = boundary[0]

    lf = min(line, key = lambda d : d['cx'] - d['w'] / 2)
    dx = abs(lf['cx'] - lf['w'] / 2 - xmin)

    if dx > LEFT_MARGIN :
        dx = abs(elem['cx'] - elem['w'] / 2 - xmin)
        if dx < LEFT_MARGIN :
            line.append(elem)
            ret = True
        else :
            for d in data[start :end] :
                dx = abs(d['cx'] - d['w'] / 2 - xmin)
                if dx < LEFT_MARGIN :
                    # iou = [calc_axis_iou(d,l,1) for l in line]
                    iou = calc_axis_iou(d, line, 1)
                    if iou > 0.3 :
                        line.append(d)
                        data.remove(d)
                        break

    return ret


def find_omit(line, content, start, end) :
    ret = False

    for i, d in enumerate(content[start :end]) :
        if calc_axis_iou(d, line, 1) > 0.3 and calc_axis_iou(d, line) < 0.01 :
            line.append(d)
            content.remove(d)

            if i == 0 :
                ret = True

    return ret


def check_wrap_title(res, wraptitles, line = None) :
    title = res[-1][0]
    wraplen = len(wraptitles)
    if wraplen :
        idx = 0
        if not line :
            wrap = wraptitles[:]
        else :
            wrap = []
            ref = min(line, key = lambda x : x['cx'] - x['w'] / 2)
            for i, w in enumerate(wraptitles) :
                if w['cy'] < ref['cy'] :
                    wrap.append(w)
                else :
                    break

        if len(wrap) :
            # if not res[-1][1]:
            #     m = re.match(r'([\*\(\)\u4e00-\u9fa5]+.*?)([\S]+?$)', title)
            #     if m:
            #         title, res[-1][1] = m.group(1), m.group(2)

            del wraptitles[0 :len(wrap)]
            title = reduce(lambda a, b : a + b, [w['text'] for w in wrap], title)
            res[-1][0] = title


def get_basic_boundary(data) :
    indexes = [i for i, d in enumerate(data) if re.search(r'开票日期|校验码', d['text'])]
    if len(indexes) :
        end = max(indexes)
    else :
        end = 8
    lt = min(data[:end + 1] + data[-10 :], key = lambda x : x['cx'] - x['w'] / 2)
    rt = max(data[:end + 1] + data[-10 :], key = lambda x : x['cx'] + x['w'] / 2)

    left = lt['cx'] - lt['w'] / 2
    right = rt['cx'] + rt['w'] / 2

    return (0, end, left, right)


def get_basic_checkcode(bg_data, fg_data) :
    checkcode = ''
    candidates = [d for d in bg_data if re.search(r'校验码*', d['text'])]
    if len(candidates) :
        checkcodes = get_candidate_after_key(candidates[0], fg_data)
        if len(checkcodes) :
            checkcode = re.sub(r'[^0-9]', '', checkcodes[0]['text'])

    return checkcode


PROVINCE = ['河北', '山西', '辽宁', '吉林', '黑龙江', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '海南', '四川', '贵州',
            '云南', '陕西']


def get_basic_type(bg_data) :
    type = ''
    title = ''
    elec = '电子' if len([d for d in bg_data if re.search(r'发票代码', d['text'])]) > 0 else ''

    candidates = [d for d in bg_data if re.search(r'.*(增?专?用?|通)?发票', d['text'])]
    if len(candidates) :
        text = candidates[0]['text']

        # if text.find('用') >= 0 or text.find('专') or text.find('增') >= 0:
        if text.find('用') >= 0 or text.find('专') >= 0 :
            type = elec + '专用发票'
        else :
            type = elec + '普通发票'

        suffix = '增值税' + type
        if text[:2] in PROVINCE :
            title = text[:2] + suffix
        elif text[:3] in PROVINCE :
            title = text[:3] + suffix
        else :
            titles = [d for d in bg_data if re.search(r'^.*增值?', d['text'])]
            if len(titles) :
                title = titles[0]['text']
                i = title.find('增')
                title = title[:i] + suffix
            else :
                i = bg_data.index(candidates[0])
                # print(len(bg_data))
                # print("i:",i)
                titles = [bg_data[i - 1], bg_data[i + 1]]
                for t in titles :
                    if re.match(r'[\u4e00-\u9fa5]{2,3}', t['text']) :
                        text = t['text']
                        if text[:2] in PROVINCE :
                            text = text[:2]
                        elif text[:3] in PROVINCE :
                            text = text[:3]
                        else :
                            text = text[:len(text) - 1]
                        title = text + suffix
                        break

    return type, title


def get_basic_title(basic, type) :
    title = ''
    elec = '电子' if len([d for d in basic if re.search(r'发票代码', d['text'])]) > 0 else ''

    candidates = [d for d in basic if re.search(r'^.*增值?', d['text'])]
    if len(candidates) :
        title = candidates[0]['text']
        i = title.find('增')
        title = title[:i] + '增值税' + elec + type

    return title


def get_basic_date(bg_data, fg_data) :
    date = ''
    candidates = [d for d in bg_data if re.search(r'开?票?日期', d['text'])]
    if len(candidates) :
        key = candidates[0]
        dates = get_candidate_after_key(key, fg_data)
        if len(dates) :
            date = re.sub(r'[^0-9年月日]', '', dates[0]['text'])
            if date[-1] != '日' :
                date += '日'

    return date


def get_basic_code(bg_data, fg_data) :
    code = ''
    candidates = [d for d in bg_data if re.search(r'发票代码', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cx'] + x['w'] / 2)
        codes = get_candidate_after_key(key, fg_data)
        if len(codes) :
            code = re.sub(r'[^0-9]', '', codes[0]['text'])
    else :
        codes = [d for d in fg_data[:10] if re.search(r'^\d{10,12}$', d['text'])]
        if len(codes) :
            code = re.sub(r'[^0-9]', '', codes[0]['text'])

    if not code :
        codes = [d for d in bg_data[:10] if re.search(r'^\d{8,}$', d['text'])]
        if len(codes) :
            code = re.sub(r'[^0-9]', '', codes[0]['text'])

    return code


def get_basic_sn(bg_data, fg_data) :
    sn = ''
    candidates = [d for d in bg_data if re.search(r'发票号码', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cx'] + x['w'] / 2)
        codes = get_candidate_after_key(key, fg_data)
        if len(codes) :
            sn = re.sub(r'[^0-9]', '', codes[0]['text'])
    else :
        codes = [d for d in fg_data[:10] if re.search(r'^\d{8}$', d['text'])]
        if len(codes) :
            sn = re.sub(r'[^0-9]', '', codes[0]['text'])

    return sn


def get_basic_person(bg_data, fg_data) :
    payee = ''
    reviewer = ''
    drawer = ''

    bg_rear = bg_data[-6 :]
    fg_rear = fg_data[-6 :]
    candidates = [d for d in bg_rear if re.search(r'收|款', d['text'])]
    if len(candidates) :
        key = candidates[0]
        payees = get_candidate_after_key(key, fg_rear)
        if len(payees) :
            payee = payees[0]['text']
            if ':' in payee :
                payee = payee.split(':')[1]

    candidates = [d for d in bg_rear if re.search(r'复|核', d['text'])]
    if len(candidates) :
        key = candidates[0]
        reviewers = get_candidate_after_key(key, fg_rear)
        if len(reviewers) :
            reviewer = reviewers[0]['text']
            if ':' in reviewer :
                reviewer = reviewer.split(':')[1]

    candidates = [d for d in bg_rear if re.search(r'开票|票人|^开$|[^款]人:', d['text'])]
    if len(candidates) :
        key = candidates[0]
        drawers = get_candidate_after_key(key, fg_rear)
        if len(drawers) :
            drawer = drawers[0]['text']
            if ':' in drawer :
                drawer = drawer.split(':')[1]

    return payee, reviewer, drawer


def getBasics(bg_data, fg_data) :
    checkcode = get_basic_checkcode(bg_data, fg_data)
    type, title = get_basic_type(bg_data)
    code = get_basic_code(bg_data, fg_data)
    sn = get_basic_sn(bg_data, fg_data)
    date = get_basic_date(bg_data, fg_data)

    payee, reviewer, drawer = get_basic_person(bg_data, fg_data)

    res = [[{'name' : r'发票类型', 'value' : type},
            {'name' : r'发票名称', 'value' : title},
            {'name' : r'发票代码', 'value' : code},
            {'name' : r'发票号码', 'value' : sn},
            {'name' : r'开票日期', 'value' : date},
            {'name' : r'校验码', 'value' : checkcode},
            {'name' : r'收款人', 'value' : payee},
            {'name' : r'复核', 'value' : reviewer},
            {'name' : r'开票人', 'value' : drawer}]]

    return res
    # return {"type":type, "title":title, "code":code, "sn":sn, "date":date, "checkcode":checkcode, "payee":payee, "reviewer":reviewer, "drawer":drawer}


def get_buyer_boundary(data) :
    indexes = [i for i, d in enumerate(data) if isTitle(d['text'])]
    end = min(indexes)

    indexes = [i for i, d in enumerate(data) if i < end and re.search(r'(开?票?日期)|(校验码)', d['text'])]
    start = max(indexes) + 1

    return start, end


def get_buyer_name(bg_data, fg_data) :
    name = ''
    candidates = [d for d in bg_data if re.search(r'名|称', d['text'])]
    if len(candidates) :
        key = candidates[0]
        names = get_candidate_after_key(key, fg_data)
        if len(names) :
            name = names[0]['text']

    return name


def get_buyer_taxnumber(bg_data, fg_data) :
    taxnumber = ''
    candidates = [d for d in bg_data if re.search(r'纳|税|人|识|别|号', d['text'])]
    if len(candidates) :
        key = candidates[0]
        taxnumbers = get_candidate_after_key(key, fg_data)
        if len(taxnumbers) :
            taxnumber = taxnumbers[0]['text']

    return taxnumber


def get_buyer_address(bg_data, fg_data) :
    address = ''
    candidates = [d for d in bg_data if re.search(r'地|址|电|话', d['text'])]
    if len(candidates) :
        key = candidates[0]
        addresses = get_candidate_after_key(key, fg_data)
        if len(addresses) :
            address = addresses[0]['text']

        if not re.search(r'[0-9\-]{11,}$', address) :
            if len(addresses) > 1 :
                number = addresses[1]
                if re.match(r'\d+$', number['text']) :
                    address += number['text']

        for prov in PROVINCE :
            idx = address.find(prov)
            if idx > 0 :
                address = address[idx :]

    return address


def get_buyer_account(bg_data, fg_data) :
    account = ''

    candidates = [d for d in bg_data if re.search(r'开|户|行|及|账号?', d['text'])]
    if len(candidates) :
        key = candidates[0]
        accounts = get_candidate_after_key(key, fg_data)
        if len(accounts) :
            account = accounts[0]['text']

        if not re.search(r'\d{12,}$', account) :
            if len(accounts) > 1 :
                number = accounts[1]
                if re.match(r'\d+$', number['text']) :
                    account += number['text']

    return account


def get_name_by_taxnum(taxnum, fg_data) :
    name = ''
    idx = [i for i, d in enumerate(fg_data) if d['text'] == taxnum]
    if len(idx) :
        idx = idx[0]
        tax = fg_data[idx]
        names = [d for d in fg_data[:idx] if calc_axis_iou(d, tax) > 0.6]
        if len(names) :
            name = names[-1]['text']

    return name


def getBuyer(bg_data, fg_data) :
    # start, end = get_buyer_boundary(bg_data)
    # bg_buyer = bg_data[start:end]
    #
    # name = get_buyer_name(bg_buyer, fg_data)
    # taxnum = get_buyer_taxnumber(bg_buyer, fg_data)
    # address = get_buyer_address(bg_buyer, fg_data)
    # account = get_buyer_account(bg_buyer, fg_data)
    #
    # if not name:
    #     name = get_name_by_taxnum(taxnum, fg_data)

    name, taxnum, address, account = ('', '', '', '')

    date = [d for d in bg_data if re.search(r'开?票?日期|校?验码', d['text'])]
    if len(date) :
        date = max(date, key = lambda x : x['cy'])
    titles = [d for d in bg_data if isTitle(d['text'])]
    for b in titles :
        if b['cy'] < date['cy'] :
            titles.remove(b)
            continue
    if len(titles) :
        title = min(titles, key = lambda x : x['cy'] - x['h'] / 2)

    candidates = [d for d in fg_data if
                  d['cy'] > date['cy'] and (d['cx'] + d['w'] / 2) < (date['cx'] - date['w'] / 2) and d['cy'] < title[
                      'cy'] and len(d['text']) > 5]
    if len(candidates) :
        sample = [c for c in candidates if re.search(r'公司|银行|学$', c['text'])]
        candidate = [c for c in candidates if calc_axis_iou(c, sample[0]) > 0.3]
        length = len(candidate)
        if length == 4 :
            candidate = sorted(candidate, key = lambda x : x['cy'])
            name, taxnum, address, account = [c['text'] for c in candidate]
        elif length == 2 :
            candidate = sorted(candidate, key = lambda x : x['cy'])
            name, taxnum = [c['text'] for c in candidate]

    res = [[{'name' : r'名称', 'value' : name},
            {'name' : r'纳税人识别号', 'value' : taxnum},
            {'name' : r'地址、电话', 'value' : address},
            {'name' : r'开户行及账号', 'value' : account}]]
    return res


def getContent(bg_data, fg_data) :
    res = []
    content = get_content_boundary(bg_data, fg_data)
    left = min(content, key = lambda x : x['cx'] - x['w'] / 2)
    right = max(content, key = lambda x : x['cx'] + x['w'] / 2)

    xmin = left['cx'] - left['w'] / 2
    xmax = right['cx'] + right['w'] / 2

    # top = min(content, key=lambda x:float(x['cy'])-float(x['h']/2))
    # bottom = max(content, key=lambda x: float(x['cy'])+float(x['h']/2))
    # lh = (bottom['cy'] + bottom['h']/2 - top['cy'] + top['h']/2) / 8

    # lt = min(content, key=lambda x:float(x['cx'])-float(x['w']/2))
    # rb = max(content, key=lambda x: float(x['cx'])+float(x['w']/2))
    # left = float(lt['cx'])-float(lt['w']/2)
    # right = float(rb['cx'])+float(rb['w']/2)

    line = []
    wraptitle = []

    for idx, ct in enumerate(content) :
        deal = False
        iswrap = is_wrapped_title(ct, line, content, idx + 1, [xmin, xmax])
        if not iswrap :
            linelen = len(line)
            if linelen :
                y_ious = []
                for l in line :
                    x_iou = calc_axis_iou(l, ct)
                    y_iou = calc_axis_iou(l, ct, 1)
                    y_ious.append(y_iou)
                    if x_iou > 0.2 :
                        deal = True
                        break
                if not deal and np.mean(y_ious) < 0.05 :
                    deal = True
                # x_iou = calc_axis_iou(ct, line)
                # y_iou = calc_axis_iou(ct, line, 1)
                # if x_iou > 0.3 or y_iou < 0.05:
                #     deal = True

            if deal == False :
                line.append(ct)
            else :
                # ret = check_title(ct, line, content, idx+1, idx+4, [xmin,xmax])
                ret = find_omit(line, content, idx, idx + 4)
                if len(res) :
                    check_wrap_title(res, wraptitle, line)
                parsed = parseLine(line, [xmin, xmax])
                res.append(parsed)

                line = []
                if not ret :
                    line.append(ct)
        else :
            wraptitle.append(ct)

    if len(line) + len(wraptitle) >= 3 :
        if len(res) :
            check_wrap_title(res, wraptitle, line)
        # if len(wraptitle):
        #     line.extend(wraptitle)
        parsed = parseLine(line, [xmin, xmax])
        res.append(parsed)

    if len(wraptitle) :
        check_wrap_title(res, wraptitle)

    ret = []
    for r in res :
        title, specs, unit, amount, uprice, price, ratio, tax = r
        # if not specs:
        #     m = re.match(r'([\*\(\)\u4e00-\u9fa5]+.*?)([\S]+?$)', title)
        #     if m:
        #         title, specs = m.group(1), m.group(2)

        if re.findall(r'[*]', title) :
            specs = specs.replace('8', 'g').replace('1', 'l')
            ret.append([{'name' : r'名称', 'value' : title},
                        {'name' : r'规格型号', 'value' : specs},
                        {'name' : r'单位', 'value' : unit},
                        {'name' : r'数量', 'value' : amount},
                        {'name' : r'单价', 'value' : uprice},
                        {'name' : r'金额', 'value' : price},
                        {'name' : r'税率', 'value' : ratio},
                        {'name' : r'税额', 'value' : tax}])
        else :
            continue

    return ret


def get_seller_boundary(data) :
    s = max([i for i, d in enumerate(data) if re.search(r'\(?大写\)?', d['text'])])
    e = min([i for i, d in enumerate(data) if re.search(r'\(?小写\)?', d['text'])])

    start = max([s, e]) + 1
    end = len(data) - 2

    return start, end


def get_seller_name(bg_data, fg_data) :
    name = ''
    candidates = [d for d in bg_data if re.search(r'名|称', d['text'])]
    if len(candidates) :
        key = candidates[0]
        names = get_candidate_after_key(key, fg_data)
        if len(names) :
            name = names[0]['text']

    return name


def get_seller_taxnumber(bg_data, fg_data) :
    taxnumber = ''
    candidates = [d for d in bg_data if re.search(r'纳|税|人|识|别|号', d['text'])]
    if len(candidates) :
        key = candidates[0]
        taxnumbers = get_candidate_after_key(key, fg_data)
        if len(taxnumbers) :
            taxnumber = taxnumbers[0]['text']

    return taxnumber


def get_seller_address(bg_data, fg_data) :
    address = ''
    candidates = [d for d in bg_data if re.search(r'地|址|电|话', d['text'])]
    if len(candidates) :
        key = candidates[0]
        addresses = get_candidate_after_key(key, fg_data)
        if len(addresses) :
            address = addresses[0]['text']

        if not re.search(r'[0-9\-]{11,}$', address) :
            if len(addresses) > 1 :
                number = addresses[1]
                if re.match(r'\d+$', number['text']) :
                    address += number['text']

        for prov in PROVINCE :
            idx = address.find(prov)
            if idx > 0 :
                address = address[idx :]

    return address


def get_seller_account(bg_data, fg_data) :
    account = ''

    candidates = [d for d in bg_data if re.search(r'开|户|行|及|账号?', d['text'])]
    if len(candidates) :
        key = candidates[0]
        accounts = get_candidate_after_key(key, fg_data)
        if len(accounts) :
            account = accounts[0]['text']

        if not re.search(r'\d{12,}$', account) :
            if len(accounts) > 1 :
                number = accounts[1]
                if re.match(r'\d+$', number['text']) :
                    account += number['text']

    return account


def merge_candidates(candidates) :
    candidates = sorted(candidates, key = lambda x : x['cy'])
    merged = []
    i = 0
    while i < len(candidates) - 1 :
        x = candidates[i]
        y = candidates[i + 1]
        if calc_axis_iou(x, y, 1) > 0.4 :
            x, y = sorted([x, y], key = lambda x : x['cx'])
            if abs(x['cx'] + x['w'] / 2 - y['cx'] + y['w'] / 2) < 20 :
                x['text'] = x['text'] + y['text']
            i += 1
        merged.append(x)
        i += 1
    if i < len(candidates) :
        merged.append(candidates[i])

    return merged


def getSeller(bg_data, fg_data) :
    # start, end = get_seller_boundary(bg_data)
    # bg_seller = bg_data[start:end]
    #
    # name = get_seller_name(bg_seller, fg_data)
    # taxnum = get_seller_taxnumber(bg_seller, fg_data)
    # address = get_seller_address(bg_seller, fg_data)
    # account = get_seller_account(bg_seller, fg_data)
    #
    # if not name:
    #     name = get_name_by_taxnum(taxnum, fg_data)

    top = [d for d in bg_data if re.search(r'\(?(价税合计|大写|小写)\)?', d['text'])]
    if len(top) :
        top = max(top, key = lambda x : x['cy'])

    candidates = [d for d in fg_data[-18 :] if d['cy'] > (top['cy'] + 15) and len(d['text']) >= 8]
    if len(candidates) :
        candidates = merge_candidates(candidates)
        sample = [c for c in candidates if re.search(r'公司|银行', c['text'])]
        candidate = [c for c in candidates if calc_axis_iou(c, sample[0]) > 0.3]
        if len(candidate) == 4 :
            candidate = sorted(candidate, key = lambda x : x['cy'])
            name, taxnum, address, account = [c['text'] for c in candidate]

    res = [[{'name' : r'名称', 'value' : name},
            {'name' : r'纳税人识别号', 'value' : taxnum},
            {'name' : r'地址、电话', 'value' : address},
            {'name' : r'开户行及账号', 'value' : account}]]
    return res
    # return {"name": name, "taxnum": taxnum, "address": address, "account": account}


def get_summation_boundary(bg_data, fg_data) :
    # keys = [d for d in bg_data if re.search(r'\(?大写\)?', d['text'])]
    # keys.extend([d for d in bg_data if re.search(r'\(?小写\)?', d['text'])])
    # keys.extend([d for d in bg_data if re.search(r'(合计?)', d['text'])])

    summation = [i for i, d in enumerate(fg_data) if re.search(r'(￥|Y|羊|F|单)\d+?\.?\d*[^\d\.]?\d*$', d['text'])]
    summdata = [fg_data[i] for i in summation]
    summation.extend([i for i, d in enumerate(fg_data) if
                      re.search(r'^\d+\.\d{2}$', d['text']) and calc_axis_iou(d, summdata, 1) > 0.4])

    keys = [d for d in bg_data if re.search(r'\(?(大写|小写)\)?', d['text'])]
    if len(keys) :
        key = max(keys, key = lambda x : x['cx'])
        summation.extend([i for i, d in enumerate(fg_data) if
                          re.search(r'^\d+\.\d{1,2}$', d['text']) and calc_axis_iou(d, key, 1) > 0.4])

    start = min(summation)
    end = max(summation) + 1

    return start, end


def getSummation(bg_data, fg_data) :
    benchmark = ['仟', '佰', '拾', '亿', '仟', '佰', '拾', '万', '仟', '佰', '拾', '圆', '角', '分']
    chinesedigit = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾']
    tax = ''
    price = ''
    total = ''
    capital = ''

    start, end = get_summation_boundary(bg_data, fg_data)
    summation = fg_data[start :end]

    prices = [re.sub(r'[^\d\.]', '', d['text']) for d in summation if
              re.search(r'(￥|Y|羊|F)?\d+?\.?\d*', d['text']) and len(d['text']) > 2]
    for i in range(len(prices)) :
        price = prices[i]
        idx = price.rfind(r'.')
        if idx <= 0 :
            price = price[:-2] + '.' + price[-2 :]
        else :
            price = price[:idx].replace(r'.', '') + (price[idx :] if len(price[idx :]) <= 3 else price[idx :idx + 3])
        prices[i] = price

    prices = list(set(prices))
    if len(prices) == 3 :
        prices = sorted(prices, key = lambda x : float(x))
        tax = prices[0]
        price = prices[1]
        total = prices[2]
        str = re.sub(r'\.', '', total)
        bm = benchmark[-len(str) :]

        for (c, b) in zip(str, bm) :
            capital += chinesedigit[int(c)] + b

        if int(total[-2 :]) == 0 :
            capital = capital[:-4] + '整'
        capital = re.sub(r'零[仟佰拾角分]', '零', capital)
        capital = re.sub(r'零{2,}', '零', capital)
        capital = re.sub(r'零$', '', capital)
        capital = re.sub(r'零圆', '圆', capital)

    res = [[{'name' : r'金额合计', 'value' : price},
            {'name' : r'税额合计', 'value' : tax},
            {'name' : r'价税合计(大写)', 'value' : capital},
            {'name' : r'价税合计(小写)', 'value' : total}]]
    return res


def calc_axis_iou(a, b, axis = 0) :
    if isinstance(b, list) :
        if len(b) :
            if axis == 0 :
                ious = [
                    calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2])
                    for x in b]
            else :
                ious = [
                    calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2])
                    for x in b]
            iou = max(ious)
        else :
            iou = 0.0
    elif isinstance(a, list) :
        if len(a) :
            if axis == 0 :
                ious = [
                    calc_iou([x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2])
                    for x in a]
            else :
                ious = [
                    calc_iou([x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2])
                    for x in a]
            iou = max(ious)
        else :
            iou = 0.0
    else :
        if axis == 0 :
            iou = calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2])
        else :
            iou = calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2])
    return iou


def get_candidate_after_key(key, data) :
    candidates = [d for d in data if d['cx'] > key['cx'] and (calc_axis_iou(d, key, 1) > 0.3 or (
                calc_axis_iou(d, key, 1) > 0.001 and -5 < (d['cx'] - d['w'] / 2 - key['cx'] - key['w'] / 2) < 30))]
    if len(candidates) :
        candidates = sorted(candidates, key = lambda x : x['cx'] - x['w'] / 2)

    return candidates


def sort_result(data) :
    data = sorted(data, key = lambda d : d['cy'])
    lines = []
    line = []
    for i in range(len(data)) :
        d = data[i]
        if not len(line) :
            line.append(d)
        else :
            iou_x = calc_axis_iou(d, line, 0)
            iou_y = calc_axis_iou(d, line, 1)
            if iou_y > 0.6 and iou_x < 0.1 :
                line.append(d)
            else :
                line = sorted(line, key = lambda l : l['cx'] - l['w'] / 2)
                lines.append(line)
                line = [d]

    if len(line) :
        line = sorted(line, key = lambda l : l['cx'] - l['w'] / 2)
        lines.append(line)

    return lines


def formatResult(bg_data, fg_data) :
    basic = getBasics(bg_data, fg_data)
    buyer = getBuyer(bg_data, fg_data)
    content = getContent(bg_data, fg_data)
    seller = getSeller(bg_data, fg_data)
    summation = getSummation(bg_data, fg_data)

    res = [{'title' : r'发票基本信息', 'items' : basic},
           {'title' : r'购买方', 'items' : buyer},
           {'title' : r'销售方', 'items' : seller},
           {'title' : r'货物或应税劳务、服务', 'items' : content},
           {'title' : r'合计', 'items' : summation}]
    return res
    # return {"basic":basic, "buyer":buyer, "content":content, "seller":seller}


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'Hyperparams')
    parser.add_argument('--arch', nargs = '?', type = str, default = 'resnet50')
    parser.add_argument('--resume', nargs = '?', type = str, default = './checkpoints/ctw1500_res50_pretrain_ic17.pth.tar',
                        help = 'Path to previous saved model to restart from')
    # parser.add_argument('--resume', nargs='?', type=str, default='/home/share/gaoluoluo/dbnet/output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth',
    #                     help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs = '?', type = float, default = 0.7,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs = '?', type = int, default = 7,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs = '?', type = int, default = 1,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs = '?', type = int, default = 2240,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs = '?', type = float, default = 5.0,
                        help = 'min kernel area')
    parser.add_argument('--min_area', nargs = '?', type = float, default = 300.0,
                        help = 'min area')
    parser.add_argument('--min_score', nargs = '?', type = float, default = 0.5,
                        help = 'min score')
    parser.add_argument('--evaluate', nargs = '?', type = bool, default = True,
                        help = 'evalution')

    args = parser.parse_args()
    test(args)