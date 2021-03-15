import os
import cv2
import sys
import json
import time
# import collections
# import torch
# import argparse
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
#import web

from flask import Flask, request, url_for, send_from_directory
from flask_cors import CORS

# from app import invoice2 as invoice
import test_app as invoice
from app import idcard_gao
from app import businesslicense_gao
from app import driverlicense_gao

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
MAX_DEAL_FILES = 6

app = Flask(__name__)
cors = CORS(app, resources = {r"/*" : {"origins" : "*"}})

# app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'data/images/tmp/')
app.config['UPLOAD_FOLDER'] = './test/test_output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def is_pdf(filename) :
    return '.' in filename and filename.rsplit('.', 1)[1] == 'pdf'


def pdf2image(path, filename):
    # return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    images = []
    pdf = fitz.open(os.path.join(path, filename))
    # 逐页读取PDF
    for pg in range(1, min(pdf.pageCount, MAX_DEAL_FILES)) :
        page = pdf[pg]
        # 设置缩放和旋转系数
        trans = fitz.Matrix(5, 5).preRotate(0)
        pm = page.getPixmap(matrix = trans, alpha = False)
        # 开始写图像
        img = os.path.join(path, str(pg) + ".png")
        pm.writePNG(img)
        images.append(img)
    pdf.close()
    return images


def image2text(params, images) :
    res = []
    recognize = None
    type = params.get('ocrtype', '')
    if type == 'invoice' :
        recognize = invoice.recognize
    elif type == 'idcard' :
        recognize = idcard_gao.recognize
    elif type == 'businesslicense':
        recognize = businesslicense_gao.recognize
    elif type == 'driverlicense' :
        recognize = driverlicense_gao.recognize

    if recognize :
        for img in images :
            result = recognize(path = img)
            res.append({"image" : os.path.basename(img), "res" : result})
    return res


# def text2excel(data, path):
#     if len(data):
#         workbook = xlwt.Workbook(encoding='utf-8')
#         sheet = workbook.add_sheet('demo')
#
#         head = ["image", "name", "text"]
#         for i in range(len(head)):
#             sheet.write(0, i, head[i])
#
#         i = 1
#         for d in data:
#             j = i
#             img = d['image']
#             for res in d['res']:
#                 # sheet.write(i, 0, img)
#                 sheet.write(i, 1, res['name'])
#                 sheet.write(i, 2, res['text'])
#                 i += 1
#             sheet.write_merge(j, i-1, 0, 0, img)
#         workbook.save(path)


@app.route('/image/<filename>', methods = ['GET', 'POST'])
def get_images(filename) :
    # filename = filename[:-4] + '_result' + filename[-4:]
    filename = filename.rsplit('.')[0] + '_result.jpg'
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods = ['GET', 'POST'])
def upload_file() :
    print("--------------------------------------------")
    t = time.time()
    res = []
    if request.method == 'POST' :
        file = request.files['file']
        params = request.values
        filename = file.filename

        if file :
            if allowed_file(filename) :
                # filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                images = [path]
                # file_url = url_for('uploaded_file', filename=filename)
            elif is_pdf(filename) :
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                images = pdf2image(path = app.config['UPLOAD_FOLDER'], filename = filename)
            res = image2text(params = params, images = images)
    timeTake = time.time() - t

    # 保存结果到xls
    # text2excel(res, './demo.xlsx')
    return json.dumps({'res' : res, 'timeTake' : round(timeTake, 4)}, ensure_ascii = False)


if __name__ == '__main__' :
    app.run(debug = False, threaded = False, host = '0.0.0.0', port = 12345)
