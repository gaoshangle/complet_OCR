#!/usr/bin/python
# encoding: utf-8
# import numpy as np
# from PIL import Image
#
# def resizeNormalize(img,imgH=32):
#         scale = img.size[1]*1.0 / imgH
#         w     = img.size[0] / scale
#         w     = int(w)
#         img   = img.resize((w,imgH),Image.BILINEAR)
#         w,h   = img.size
#         img = (np.array(img)/255.0-0.5)/0.5
#         return img
#
#
# def strLabelConverter(res,alphabet):
#         N = len(res)
#         raw = []
#         for i in range(N):
#             if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
#                 raw.append(alphabet[res[i] - 1])
#         return ''.join(raw)

import numpy as np
from PIL import Image


def resizeNormalize(img, imgH = 32) :
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), Image.BILINEAR)
        w, h = img.size
        img = (np.array(img) / 255.0 - 0.5) / 0.5
        return img


def strLabelConverter(res, alphabet) :
        N = len(res)
        raw = []
        same_words = []
        for i in range(N) :
                if res[i][0][0] != 0 and (not (i > 0 and res[i - 1][0][0] == res[i][0][0])) :
                        raw.append(alphabet[res[i][0][0] - 1])
                        same_word = []
                        flag = True
                        for k in range(len(res[i][0])) :
                                if (res[i][0][k] != 0 and flag) :
                                        if k > 0 :
                                                same_word.append(alphabet[res[i][0][k] - 1])
                                else :
                                        flag = False
                        same_words.append(same_word)
        return [''.join(raw), same_words]
    
