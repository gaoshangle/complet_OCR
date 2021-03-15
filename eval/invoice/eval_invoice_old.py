#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: pfq time: 2020/8/29 0029
import numpy as np
import re
import os
from apphelper.image import calc_iou, solve, order_point


def calc_axis_iou(a, b, axis = 0) :
    if isinstance(b, list) :
        if axis == 0 :
            ious = [calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2])
                    for x in b]
        else :
            ious = [calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2])
                    for x in b]
        iou = max(ious)
    elif isinstance(a, list) :
        if axis == 0 :
            ious = [calc_iou([x['cx'] - x['w'] / 2, x['cx'] + x['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2])
                    for x in a]
        else :
            ious = [calc_iou([x['cy'] - x['h'] / 2, x['cy'] + x['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2])
                    for x in a]
        iou = max(ious)
    else :
        if axis == 0 :
            iou = calc_iou([a['cx'] - a['w'] / 2, a['cx'] + a['w'] / 2], [b['cx'] - b['w'] / 2, b['cx'] + b['w'] / 2])
        else :
            iou = calc_iou([a['cy'] - a['h'] / 2, a['cy'] + a['h'] / 2], [b['cy'] - b['h'] / 2, b['cy'] + b['h'] / 2])
    return iou


def calc_distance(a, b) :
    dx = abs(a['cx'] - a['w'] / 2 - b['cx'] - b['w'] / 2)
    return dx


def get_candidate_after_key(key, data) :
    candidates = [d for d in data if calc_axis_iou(d, key, 1) > 0.15 and d['cx'] > key['cx']]
    if len(candidates) :
        candidates = sorted(candidates, key = lambda x : x['cx'] - x['w'] / 2)
        if len(candidates) > 1 and calc_axis_iou(candidates[0], candidates[1 :]) > 0.3 :
            ref = candidates[0]
            candidates = [c for c in candidates if calc_axis_iou(c, ref) > 0.3]
            # candidates = sorted(candidates, key=lambda x:calc_axis_iou(x,key,1), reverse=True)
    return candidates


def get_index_after_key(key, data) :
    index = [i for i, d in enumerate(data) if calc_axis_iou(d, key, 1) > 0.15 and d['cx'] > key['cx']]
    if len(index) :
        index = sorted(index, key = lambda i : data[i]['cx'] - data[i]['w'] / 2)
        if len(index) > 1 :
            ref = index[0]
            index = [i for i in index if calc_axis_iou(data[i], data[ref]) > 0.3]
            # candidates = sorted(candidates, key=lambda x:calc_axis_iou(x,key,1), reverse=True)
    return index


def is_title(text) :
    return text.find(r'服务名称') > -1 \
            or text.find(r'规格型号') > -1 \
            or re.fullmatch(r'单位', text) \
            or text.find(r'数量') > -1 \
            or text.find(r'单价') > -1 \
            or text.find(r'金额') > -1 \
            or text.find(r'税率') > -1 \
            or text.find(r'税额') > -1 \
            or text.find(r'项目名称') > -1 \
            or text.find(r'车牌号') > -1 \
            or re.fullmatch(r'类型', text) \
            or text.find(r'通行日期起') > -1 \
            or text.find(r'通行日期止') > -1 and text.find(r'[*]') == -1

    # or text.find(r'单位') > -1 \


def get_basic_checkcode(data) :
    checkcode = ''
    candidates = [d for d in data if re.search(r'校验码*', d['text'])]
    if len(candidates) :
        checkcodes = get_candidate_after_key(candidates[0], data)
        if len(checkcodes) :
            checkcode = checkcodes[0]['text']

    return checkcode


def get_basic_type(data) :
    type = ''
    title = ''
    elec = '电子' if len([d for d in data if re.search(r'发票代码', d['text'])]) > 0 else ''

    candidates = [d for d in data if re.search(r'.*增值税.*发票', d['text'])]
    if len(candidates) :
        title = candidates[0]['text']

        if title.find('专用') >= 0 :
            type = elec + '专用发票'
        else :
            type = elec + '普通发票'

    return type, title


def get_basic_code(data) :
    code = ''
    candidates = [d for d in data if re.search(r'发票代码', d['text'])]
    if len(candidates) :
        key = candidates[0]
        codes = get_candidate_after_key(key, data)
        if len(codes) :
            code = codes[0]['text']
    else :
        codes = [d for d in data[:6] if re.search(r'^\d{10,12}$', d['text'])]
        if len(codes) :
            code = codes[0]['text']

    return code


def get_basic_sn(data) :
    sn = ''
    candidates = [d for d in data if re.search(r'发票号码', d['text'])]
    if len(candidates) :
        key = candidates[0]
        codes = get_candidate_after_key(key, data)
        if len(codes) :
            sn = codes[0]['text']
    else :
        codes = [d for d in data[:10] if re.search(r'^\d{8}$', d['text'])]
        if len(codes) :
            sn = codes[0]['text']

    return sn


def get_basic_date(data) :
    date = ''
    candidates = [d for d in data if re.search(r'开票日期', d['text'])]
    if len(candidates) :
        key = candidates[0]
        dates = get_candidate_after_key(key, data)
        if len(dates) :
            date = dates[0]['text']

    return date


def get_basic_person(data) :
    payee = ''
    reviewer = ''
    drawer = ''

    rear = data
    candidates = [d for d in rear if re.search(r'收款人', d['text'])]
    if len(candidates) :
        key = candidates[0]
        payees = get_candidate_after_key(key, rear)
        # if len(payees) and calc_distance(payees[0], key) < 30 :
        if len(payees) and calc_distance(payees[0], key) < 35 :
            payee = payees[0]['text']

    candidates = [d for d in rear if re.search(r'复核', d['text'])]
    if len(candidates) :
        key = candidates[0]
        reviewers = get_candidate_after_key(key, rear)
        # if len(reviewers) and calc_distance(reviewers[0], key) < 30 :
        if len(reviewers) and calc_distance(reviewers[0], key) < 35 :
            reviewer = reviewers[0]['text']

    candidates = [d for d in rear if re.search(r'开票人', d['text'])]
    if len(candidates) :
        key = candidates[0]
        drawers = get_candidate_after_key(key, rear)
        if len(drawers) and calc_distance(drawers[0], key) < 30 :
            drawer = drawers[0]['text']

    return payee, reviewer, drawer


def getBasics(gt) :
    checkcode = get_basic_checkcode(gt)
    type, title = get_basic_type(gt)
    code = get_basic_code(gt)
    sn = get_basic_sn(gt)
    date = get_basic_date(gt)

    payee, reviewer, drawer = get_basic_person(gt)

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


def getBuyer(data) :
    name, taxnum, address, account = ('', '', '', '')
    front = data
    candidates = [d for d in front if re.search(r'名称', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        names = get_candidate_after_key(key, front)
        if len(names) :
            name = names[0]['text']

    candidates = [d for d in front if re.search(r'纳税人识别号', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        taxnums = get_candidate_after_key(key, front)
        if len(taxnums) :
            taxnum = taxnums[0]['text']

    candidates = [d for d in front if re.search(r'地址、电话', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        addresses = get_candidate_after_key(key, front)
        if len(addresses) :
            address = addresses[0]
            if abs(key['cx'] + key['w'] / 2 - address['cx'] + address['w'] / 2) < 100 :
                address = address['text']
            else :
                address = ''

    candidates = [d for d in front if re.search(r'开户行及账号', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        accounts = get_candidate_after_key(key, front)
        if len(accounts) :
            account = accounts[0]
            if abs(key['cx'] + key['w'] / 2 - account['cx'] + account['w'] / 2) < 100 :
                account = account['text']
            else :
                account = ''

    res = [[{'name' : r'名称', 'value' : name},
            {'name' : r'纳税人识别号', 'value' : taxnum},
            {'name' : r'地址、电话', 'value' : address},
            {'name' : r'开户行及账号', 'value' : account}]]
    return res


def get_content_boundary(data) :
    titles = [d for d in data if is_title(d['text'])]
    right = max(titles, key = lambda x : x['cx'] + x['w'] / 2)
    bottom = min(titles, key = lambda x : x['cy'] + x['h'] / 2)

    summary = [d for d in data if re.search(r'￥\d+', d['text'])]
    summary = min(summary, key = lambda x : x['cy'] - x['h'] / 2)

    content = [d for d in data if
               d['cy'] > (bottom['cy'] + bottom['h'] / 2) and d['cy'] < (
                           summary['cy'] - summary['h'] / 2) and not is_title(d['text'])]
    return content


def doLeft(content, item, length) :
    if len(item) < length :
        ct = [c for c in content if calc_axis_iou(c, item) > 0]
        if len(ct) + len(item) == length :
            item = sorted(item + ct, key = lambda x : x['cy'])
            for c in ct :
                content.remove(c)

    return item, content


def getContent(data) :
    ret = []

    content = get_content_boundary(data)
    titles = [d for d in data if is_title(d['text'])]
    titles = sorted(titles, key = lambda x : x['cx'])
    s = re.compile(r'[-,$()#+&*~]')
    for i in range(len(titles)) :
        if re.findall(s, titles[i]['text']) or re.findall(r'[a-zA-Z0-9]', titles[i]['text']) :
            del titles[i]
            break

    t_name, t_spec, t_unit, t_num, t_uprice, t_price, t_ratio, t_tax = titles

    isvehicle = (t_name == '项目名称')

    taxes = sorted([c for c in content if calc_axis_iou(c, t_tax) > 0.001 or c['cx'] >= t_tax['cx']],
                   key = lambda x : x['cy'])
    names = sorted([c for c in content if calc_axis_iou(c, t_name) > 0.001], key = lambda x : x['cy'])
    ratios = sorted([c for c in content if calc_axis_iou(c, t_ratio) > 0.001], key = lambda x : x['cy'])
    prices = sorted([c for c in content if calc_axis_iou(c, t_price) > 0.001 or (
                c['cx'] + c['w'] / 2 < t_ratio['cx'] - t_ratio['w'] / 2 and c['cx'] - c['w'] / 2 > t_price['cx'] +
                t_price['w'] / 2)], key = lambda x : x['cy'])
    uprices = sorted([c for c in content if calc_axis_iou(c, t_uprice) > 0.001 or (
                c['cx'] + c['w'] / 2 < t_price['cx'] - t_price['w'] / 2 and c['cx'] - c['w'] / 2 > t_uprice['cx'] +
                t_uprice['w'] / 2)], key = lambda x : x['cy'])
    nums = sorted([c for c in content if calc_axis_iou(c, t_num) > 0.001 or (
                c['cx'] + c['w'] / 2 < t_uprice['cx'] - t_uprice['w'] / 2 and c['cx'] - c['w'] / 2 > t_unit['cx'] +
                t_unit['w'] / 2)], key = lambda x : x['cy'])
    units = sorted([c for c in content if calc_axis_iou(c, t_unit) > 0.001], key = lambda x : x['cy'])
    specs = sorted([c for c in content if calc_axis_iou(c, t_spec) > 0.001 or (
                c['cx'] + c['w'] / 2 < t_spec['cx'] - t_spec['w'] / 2 and c['cx'] - c['w'] / 2 > t_name['cx'] + t_name[
            'w'] / 2)], key = lambda x : x['cy'])

    done = taxes + names + ratios + prices + uprices + nums + units + specs
    left = [c for c in content if c not in done]
    if len(left) :
        specs, left = doLeft(left, specs, len(taxes))
    if len(left) :
        units, left = doLeft(left, units, len(taxes))
    if len(left) :
        nums, left = doLeft(left, nums, len(taxes))

    if len(taxes) != len(names) :
        merges = []
        idx = [i for i, n in enumerate(names) if re.search(r'^\*', n['text'])]
        j = 0
        for i in idx[1 :] :
            merge = names[j]
            merge['text'] = ''.join([n['text'] for n in names[j :i]])
            merges.append(merge)
            j = i
        if j < len(names) :
            merge = names[j]
            merge['text'] = ''.join([n['text'] for n in names[j :]])
            merges.append(merge)
        names = merges

    for i in range(len(taxes)) :
        name = names[i]['text']
        tax = taxes[i]['text']
        ratio = ratios[i]['text']
        price = prices[i]['text']
        uprice = uprices[i]['text'] if i < len(uprices) else ''
        num = nums[i]['text'] if i < len(nums) else ''
        unit = units[i]['text'] if i < len(units) else ''
        spec = specs[i]['text'] if i < len(specs) else ''

        if isvehicle :
            ret.append([{'name' : r'项目名称', 'value' : name},
                        {'name' : r'车牌号', 'value' : spec},
                        {'name' : r'类型', 'value' : unit},
                        {'name' : r'通行日期起', 'value' : num},
                        {'name' : r'通行日期止', 'value' : uprice},
                        {'name' : r'金额', 'value' : price},
                        {'name' : r'税率', 'value' : ratio},
                        {'name' : r'税额', 'value' : tax}])
        else :
            ret.append([{'name' : r'名称', 'value' : name},
                        {'name' : r'规格型号', 'value' : spec},
                        {'name' : r'单位', 'value' : unit},
                        {'name' : r'数量', 'value' : num},
                        {'name' : r'单价', 'value' : uprice},
                        {'name' : r'金额', 'value' : price},
                        {'name' : r'税率', 'value' : ratio},
                        {'name' : r'税额', 'value' : tax}])
    return ret


def getSeller(data) :
    name, taxnum, address, account = ('', '', '', '')

    rear = data
    candidates = [d for d in rear if re.search(r'名称', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # names = get_candidate_after_key(key, rear)
        # if len(names):
        #     name = names[0]['text']
        names = get_index_after_key(key, rear)

    candidates = [d for d in rear if re.search(r'纳税人识别号', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # taxnums = get_candidate_after_key(key, rear)
        # if len(taxnums):
        #     taxnum = taxnums[0]['text']
        taxnums = get_index_after_key(key, rear)

    candidates = [d for d in rear if re.search(r'地址、电话', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # addresses = get_candidate_after_key(key, rear)
        # if len(addresses):
        #     address = addresses[0]['text']
        addresses = get_index_after_key(key, rear)

    candidates = [d for d in rear if re.search(r'开户行及账号', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # accounts = get_candidate_after_key(key, rear)
        # if len(accounts):
        #     account = accounts[0]['text']
        accounts = get_index_after_key(key, rear)

    candidates = list(set(names + taxnums + addresses + accounts))
    candidates = sorted(candidates, key = lambda x : rear[x]['cy'])
    length = len(candidates)
    if length >= 4 :
        name, taxnum, address, account = [rear[c]['text'] for c in candidates[0 :4]]
    elif length == 3 :
        name, taxnum, address = [rear[c]['text'] for c in candidates]
        account = ''
    elif length == 2 :
        name, taxnum = [rear[c]['text'] for c in candidates]
        address = ''
        account = ''
    elif length == 1 :
        name = candidates[0]['text']
        taxnum = '',
        address = ''
        account = ''
    else :
        name = ''
        taxnum = '',
        address = ''
        account = ''

    res = [[{'name' : r'名称', 'value' : name},
            {'name' : r'纳税人识别号', 'value' : taxnum},
            {'name' : r'地址、电话', 'value' : address},
            {'name' : r'开户行及账号', 'value' : account}]]
    return res


def getSummation(data) :
    prices = [re.sub(r'[^\d\.]', '', d['text']) for d in data if re.search(r'￥\d+\.\d{1,2}', d['text'])]
    prices = sorted(prices, key = lambda x : float(x))
    if len(prices) == 2 :
        tax = '***'
        price = prices[0]
        total = prices[1]
    else :
        tax = prices[0]
        price = prices[1]
        total = prices[2]

    candidates = [d for d in data if re.search(r'价税合计', d['text'])]
    if len(candidates) :
        key = candidates[0]
        capitals = get_candidate_after_key(key, data)
        if len(capitals) :
            capital = capitals[0]['text']

    res = [[{'name' : r'金额合计', 'value' : price},
            {'name' : r'税额合计', 'value' : tax},
            {'name' : r'价税合计(大写)', 'value' : capital},
            {'name' : r'价税合计(小写)', 'value' : total}]]
    return res


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


def calc_precision1(predict, groundtruth, errfile) :    # 以字符方式计算正确率
    total = 0
    error = 0

    with open(errfile, 'w', encoding = 'utf-8') as f :
        for (p, g) in zip(predict, groundtruth) :
            for (pi, gi) in zip(p['items'], g['items']) :
                for (pii, gii) in zip(pi, gi) :
                    distance = get_min_distance(pii['value'], gii['value'])
                    error += distance
                    total += len(gii['value'])

                    if distance :
                        text = p['title'] + ' ' + pii['name'] + ' ' + 'groundtruth: ' + gii['value'] + ' errortext: ' + \
                               pii['value'] + '\n'
                        f.write(text)

    precision = round(100.0 * (total - error) / total, 2)
    return precision

def calc_precision2(gt_len, errfile) :    # 以字段计算正确率
    with open(errfile, 'r', encoding = 'utf-8') as f :
        err_len = len(f.readlines())

    precision = round(100.0 * (gt_len - err_len) / gt_len, 2)
    return precision


def genGT(file) :
    with open(file, 'r', encoding = 'utf-8') as f :
        gt = []
        lines = f.readlines()
        for line in lines :
            line = line.split(' ')
            points = line[0]
            text = re.sub(r'\n', '', ''.join(line[1 :]))
            box = [float(p) for p in points.split(',')]
            box = np.array(box).astype('int32')
            box = box.reshape(4, 2)
            box = order_point(box)
            box = box.reshape(-1)
            angle, w, h, cx, cy = solve(box)
            gt.append({'angle' : angle, 'w' : w, 'h' : h, 'cx' : cx, 'cy' : cy, 'text' : text})

        basic = getBasics(gt)
        buyer = getBuyer(gt)
        content = getContent(gt)
        seller = getSeller(gt)
        summation = getSummation(gt)

        groundtruth = [{'title' : r'发票基本信息', 'items' : basic},
                       {'title' : r'购买方', 'items' : buyer},
                       {'title' : r'销售方', 'items' : seller},
                       {'title' : r'货物或应税劳务、服务', 'items' : content},
                       {'title' : r'合计', 'items' : summation}]

    return groundtruth


def evaluate(file, errfile, predict = None) :
    with open(file, 'r', encoding = 'utf-8') as f :
        gt = []
        precision = []
        lines = f.readlines()
        for line in lines :
            line = line.split(' ')
            points = line[0]
            text = re.sub(r'\n', '', ''.join(line[1 :]))
            box = [float(p) for p in points.split(',')]
            box = np.array(box).astype('int32')
            box = box.reshape(4, 2)
            box = order_point(box)
            box = box.reshape(-1)
            angle, w, h, cx, cy = solve(box)
            gt.append({'angle' : angle, 'w' : w, 'h' : h, 'cx' : cx, 'cy' : cy, 'text' : text})

        basic = getBasics(gt)
        buyer = getBuyer(gt)
        content = getContent(gt)
        seller = getSeller(gt)
        summation = getSummation(gt)

        gt_len = len(gt)

        groundtruth = [{'title' : r'发票基本信息', 'items' : basic},
                       {'title' : r'购买方', 'items' : buyer},
                       {'title' : r'销售方', 'items' : seller},
                       {'title' : r'货物或应税劳务、服务', 'items' : content},
                       {'title' : r'合计', 'items' : summation}]

        precision1 = calc_precision1(predict, groundtruth, errfile)
        precision2 = calc_precision2(gt_len, errfile)
        precision.append(precision1)
        precision.append(precision2)
        return precision


if __name__ == '__main__' :
    image_name = 'baidu-3'
    image_file = image_name + '.txt'
    error_file = image_name + '-errors.txt'
    file = os.path.join(os.path.dirname(__file__), '../data/gt/', image_file)
    genGT(file)