import numpy as np
import re
import os
from apphelper.image import calc_iou, solve, order_point
# from correction import corrector


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
    return text.find(r'????????????') > -1 \
           or text.find(r'????????????') > -1 \
           or re.fullmatch(r'??????', text) \
           or text.find(r'??????') > -1 \
           or text.find(r'??????') > -1 \
           or text.find(r'??????') > -1 \
           or text.find(r'??????') > -1 \
           or text.find(r'??????') > -1 \
           or text.find(r'????????????') > -1 \
           or text.find(r'?????????') > -1 \
           or re.fullmatch(r'??????', text) \
           or text.find(r'???????????????') > -1 \
           or text.find(r'???????????????') > -1 and text.find(r'[*]') == -1

    # or text.find(r'??????') > -1 \


def get_basic_checkcode(data) :
    checkcode = ''
    candidates = [d for d in data if re.search(r'?????????*', d['text'])]
    if len(candidates) :
        checkcodes = get_candidate_after_key(candidates[0], data)
        if len(checkcodes) :
            checkcode = checkcodes[0]['text']

    return checkcode


def get_basic_type(data) :
    type = ''
    title = ''
    elec = '??????' if len([d for d in data if re.search(r'????????????', d['text'])]) > 0 else ''

    candidates = [d for d in data if re.search(r'.*?????????.*??????', d['text'])]
    if len(candidates) :
        title = candidates[0]['text']

        if title.find('??????') >= 0 :
            type = elec + '????????????'
        else :
            type = elec + '????????????'

    return type, title


def get_basic_code(data) :
    code = ''
    candidates = [d for d in data if re.search(r'????????????', d['text'])]
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
    candidates = [d for d in data if re.search(r'????????????', d['text'])]
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
    candidates = [d for d in data if re.search(r'????????????', d['text'])]
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
    candidates = [d for d in rear if re.search(r'?????????', d['text'])]
    if len(candidates) :
        key = candidates[0]
        payees = get_candidate_after_key(key, rear)
        # if len(payees) and calc_distance(payees[0], key) < 30 :
        if len(payees) and calc_distance(payees[0], key) < 35 :
            payee = payees[0]['text']

    candidates = [d for d in rear if re.search(r'??????', d['text'])]
    if len(candidates) :
        key = candidates[0]
        reviewers = get_candidate_after_key(key, rear)
        # if len(reviewers) and calc_distance(reviewers[0], key) < 30 :
        if len(reviewers) and calc_distance(reviewers[0], key) < 35 :
            reviewer = reviewers[0]['text']

    candidates = [d for d in rear if re.search(r'?????????', d['text'])]
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

    res = [[{'name' : r'????????????', 'value' : type},
            {'name' : r'????????????', 'value' : title},
            {'name' : r'????????????', 'value' : code},
            {'name' : r'????????????', 'value' : sn},
            {'name' : r'????????????', 'value' : date},
            {'name' : r'?????????', 'value' : checkcode},
            {'name' : r'?????????', 'value' : payee},
            {'name' : r'??????', 'value' : reviewer},
            {'name' : r'?????????', 'value' : drawer}]]

    return res


def getBuyer(data) :
    name, taxnum, address, account = ('', '', '', '')
    front = data
    candidates = [d for d in front if re.search(r'??????', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        names = get_candidate_after_key(key, front)
        if len(names) :
            name = names[0]['text']

    candidates = [d for d in front if re.search(r'??????????????????', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        taxnums = get_candidate_after_key(key, front)
        if len(taxnums) :
            taxnum = taxnums[0]['text']

    candidates = [d for d in front if re.search(r'???????????????', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        addresses = get_candidate_after_key(key, front)
        if len(addresses) :
            address = addresses[0]
            if abs(key['cx'] + key['w'] / 2 - address['cx'] + address['w'] / 2) < 100 :
                address = address['text']
            else :
                address = ''

    candidates = [d for d in front if re.search(r'??????????????????', d['text'])]
    if len(candidates) :
        key = min(candidates, key = lambda x : x['cy'])
        accounts = get_candidate_after_key(key, front)
        if len(accounts) :
            account = accounts[0]
            if abs(key['cx'] + key['w'] / 2 - account['cx'] + account['w'] / 2) < 100 :
                account = account['text']
            else :
                account = ''

    res = [[{'name' : r'??????', 'value' : name},
            {'name' : r'??????????????????', 'value' : taxnum},
            {'name' : r'???????????????', 'value' : address},
            {'name' : r'??????????????????', 'value' : account}]]
    return res


def get_content_boundary(data) :
    titles = [d for d in data if is_title(d['text'])]
    right = max(titles, key = lambda x : x['cx'] + x['w'] / 2)
    bottom = min(titles, key = lambda x : x['cy'] + x['h'] / 2)

    summary = [d for d in data if re.search(r'???\d+', d['text'])]
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

    isvehicle = (t_name == '????????????')
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
            ret.append([{'name' : r'????????????', 'value' : name},
                        {'name' : r'?????????', 'value' : spec},
                        {'name' : r'??????', 'value' : unit},
                        {'name' : r'???????????????', 'value' : num},
                        {'name' : r'???????????????', 'value' : uprice},
                        {'name' : r'??????', 'value' : price},
                        {'name' : r'??????', 'value' : ratio},
                        {'name' : r'??????', 'value' : tax}])
        else :
            ret.append([{'name' : r'??????', 'value' : name},
                        {'name' : r'????????????', 'value' : spec},
                        {'name' : r'??????', 'value' : unit},
                        {'name' : r'??????', 'value' : num},
                        {'name' : r'??????', 'value' : uprice},
                        {'name' : r'??????', 'value' : price},
                        {'name' : r'??????', 'value' : ratio},
                        {'name' : r'??????', 'value' : tax}])
    return ret


def getSeller(data) :
    name, taxnum, address, account = ('', '', '', '')

    rear = data
    candidates = [d for d in rear if re.search(r'??????', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # names = get_candidate_after_key(key, rear)
        # if len(names):
        #     name = names[0]['text']
        names = get_index_after_key(key, rear)

    candidates = [d for d in rear if re.search(r'??????????????????', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # taxnums = get_candidate_after_key(key, rear)
        # if len(taxnums):
        #     taxnum = taxnums[0]['text']
        taxnums = get_index_after_key(key, rear)

    candidates = [d for d in rear if re.search(r'???????????????', d['text'])]
    if len(candidates) :
        key = max(candidates, key = lambda x : x['cy'])
        # addresses = get_candidate_after_key(key, rear)
        # if len(addresses):
        #     address = addresses[0]['text']
        addresses = get_index_after_key(key, rear)

    candidates = [d for d in rear if re.search(r'??????????????????', d['text'])]
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

    res = [[{'name' : r'??????', 'value' : name},
            {'name' : r'??????????????????', 'value' : taxnum},
            {'name' : r'???????????????', 'value' : address},
            {'name' : r'??????????????????', 'value' : account}]]
    return res


def getSummation(data) :
    prices = [re.sub(r'[^\d\.]', '', d['text']) for d in data if re.search(r'???\d+\.\d{1,2}', d['text'])]
    prices = sorted(prices, key = lambda x : float(x))
    if len(prices) == 2 :
        tax = '***'
        price = prices[0]
        total = prices[1]
    else :
        tax = prices[0]
        price = prices[1]
        total = prices[2]

    candidates = [d for d in data if re.search(r'????????????', d['text'])]
    if len(candidates) :
        key = candidates[0]
        capitals = get_candidate_after_key(key, data)
        if len(capitals) :
            capital = capitals[0]['text']

    res = [[{'name' : r'????????????', 'value' : price},
            {'name' : r'????????????', 'value' : tax},
            {'name' : r'????????????(??????)', 'value' : capital},
            {'name' : r'????????????(??????)', 'value' : total}]]
    return res


def get_min_distance(word1, word2) :
    m, n = len(word1), len(word2)
    if m == 0 : return n
    if n == 0 : return m
    cur = [0] * (m + 1)  # ?????????cur?????????
    for i in range(1, m + 1) : cur[i] = i

    for j in range(1, n + 1) :  # ??????cur
        pre, cur[0] = cur[0], j  # ?????????????????????????????????
        for i in range(1, m + 1) :
            temp = cur[i]  # ?????????????????????????????????
            if word1[i - 1] == word2[j - 1] :
                cur[i] = pre
            else :
                cur[i] = min(pre + 1, cur[i] + 1, cur[i - 1] + 1)
            pre = temp
    return cur[m]


def calc_precision1(predict, groundtruth, errfile, canfile) :  # ??????????????????????????????

    # total = 0
    # error = 0
    #
    # with open(errfile, 'w', encoding = 'utf-8') as f :
    #     for (p, g) in zip(predict, groundtruth) :
    #         for (pi, gi) in zip(p['items'], g['items']) :
    #             for (pii, gii) in zip(pi, gi) :
    #                 distance = get_min_distance(pii['value'], gii['value'])
    #                 error += distance
    #                 total += len(gii['value'])
    #
    #                 if distance :
    #                     text = p['title'] + ' ' + pii['name'] + ' ' + 'groundtruth: ' + gii['value'] + ' errortext: ' + \
    #                            pii['value'] + '\n'
    #                     f.write(text)
    #
    # precision = round(100.0 * (total - error) / total, 2)
    # return precision
    total = 0
    error = 0

    with open(errfile, 'w', encoding = 'utf-8') as f :
        for (p, g) in zip(predict, groundtruth) :
            for (pi, gi) in zip(p['items'], g['items']) :
                for (pii, gii) in zip(pi, gi) :
                    distance = get_min_distance(pii['value'], gii['value'])
                    if distance and ("??????" in pii['value'] and "???" in pii['value']) :
                        error_sen = pii['value']
                        # pii['value'] = corrector.get_corrected(error_sen)
                        if 'I' in pii['value'] :
                            pii['value'] = pii['value'].replace('I', '1')
                        if 'l' in pii['value'] :
                            pii['value'] = pii['value'].replace('l', '1')
                        distance = get_min_distance(pii['value'], gii['value'])
                    error += distance
                    total += len(gii['value'])

                    if distance :
                        text = p['title'] + ' ' + pii['name'] + ' ' + 'groundtruth: ' + gii['value'] + ' errortext: ' + \
                               pii['value'] + '\n'
                        f.write(text)
    with open(canfile, 'w', encoding = 'utf-8') as fc :
        for (p, g) in zip(predict, groundtruth) :
            for (pi, gi) in zip(p['items'], g['items']) :
                for (pii, gii) in zip(pi, gi) :
                    #                                                                                                        ????????????????????????candidate????????????name???value???
                    text = (gii['value'] if gii['value'] else '-') + '\t' + (pii['value'] if pii['value'] else '-') + '\n'
                    # text = (gii['value'] if gii['value'] else '-') + '\t' + (pii['value'] if pii['value'] else '-') + '\t' + str(pii['candidate']) + '\n'
                    fc.write(text)
    fc.close()
    precision = round(100.0 * (total - error) / total, 2)
    return precision


def calc_precision2(gt_len, errfile) :  # ????????????????????????
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

        groundtruth = [{'title' : r'??????????????????', 'items' : basic},
                       {'title' : r'?????????', 'items' : buyer},
                       {'title' : r'?????????', 'items' : seller},
                       {'title' : r'??????????????????????????????', 'items' : content},
                       {'title' : r'??????', 'items' : summation}]

    return groundtruth


def evaluate(file, errfile, canfile, predict = None) :
    with open(file, 'r', encoding = 'utf-8') as f :
        gt = []
        precision = []
        lines = f.readlines()
        # print("file:",file)
        # print("len:",len(lines))
        # i = 1
        for line in lines :
            line = line.split(',')

            flag = len(line)

            if flag==8:# ??????
                points = line[:7]
                line = line[7].split(' ')
                points.append(line[0])
                text = ''
                for i in range(1,len(line)):
                    text += line[i]
                text = re.sub(r'\n', '', ''.join(text))
                # print("points:",points)
                box = [float(p) for p in points]
            else:#??????
                points = line[:8]
                text = line[8:]
                text = re.sub(r'\n','',''.join(text))
                box = [float(p) for p in points]
                pass
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

        groundtruth = [{'title' : r'??????????????????', 'items' : basic},
                       {'title' : r'?????????', 'items' : buyer},
                       {'title' : r'?????????', 'items' : seller},
                       {'title' : r'??????????????????????????????', 'items' : content},
                       {'title' : r'??????', 'items' : summation}]
        precision1 = calc_precision1(predict, groundtruth, errfile, canfile)
        # precision2 = calc_precision2(gt_len, errfile) # ??????
        precision.append(precision1)
        # precision.append(precision2)
        return precision


if __name__ == '__main__' :
    image_name = 'baidu-3'
    image_file = image_name + '.txt'
    error_file = image_name + '-errors.txt'
    file = os.path.join(os.path.dirname(__file__), '../data/gt/', image_file)
    genGT(file)
