# import os
# import subprocess
# import cv2
# import numpy as np
# from PIL import Image
# from config import *
# from crnn.keys import alphabetChinese,alphabetEnglish
# from apphelper.image import rotate_cut_img,sort_box,union_rbox
#
# if ocrFlag=='keras':
#     from crnn.network_keras import CRNN
#     if chineseModel:
#         alphabet = alphabetChinese
#         if LSTMFLAG:
#             ocrModel = ocrModelKerasLstm
#         else:
#             ocrModel = ocrModelKerasDense
#     else:
#         ocrModel = ocrModelKerasEng
#         alphabet = alphabetEnglish
#         LSTMFLAG = True
# elif ocrFlag=='torch':
#     from crnn.network_torch import CRNN
#     if chineseModel:
#         alphabet = alphabetChinese
#         if LSTMFLAG:
#             ocrModel = ocrModelTorchLstm
#         else:
#             ocrModel = ocrModelTorchDense
#
#     else:
#         ocrModel = ocrModelTorchEng
#         alphabet = alphabetEnglish
#         LSTMFLAG = True
# elif ocrFlag=='opencv':
#     from crnn.network_dnn import CRNN
#     ocrModel = ocrModelOpencv
#     alphabet = alphabetChinese
# else:
#     print( "err,ocr engine in keras\opencv\darknet")
#
# nclass = len(alphabet)+1
# if ocrFlag=='opencv':
#     crnn = CRNN(alphabet=alphabet)
# else:
#     crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
# if os.path.exists(ocrModel):
#     crnn.load_weights(ocrModel)
# else:
#     print("download model or tranform model with tools!")
#
# recognizer = crnn.predict
#
# def crnnRec(im, boxes, leftAdjust=False, rightAdjust=False, alph=0.2, f=1.0):
#     """
#     crnn模型，ocr识别
#     @@model,
#     @@converter,
#     @@im:Array
#     @@text_recs:text box
#     @@ifIm:是否输出box对应的img
#
#     """
#     results = []
#
#     im = Image.fromarray(im)
#     boxes = sort_box(boxes)
#     for index, box in enumerate(boxes):
#
#         degree, w, h, cx, cy = box
#
#         # partImg, newW, newH = rotate_cut_img(im,  90  + degree  , cx, cy, w, h, leftAdjust, rightAdjust, alph)
#         partImg = crop_rect(im, ((cx, cy), (h, w), degree))
#         newW, newH = partImg.size
#         # partImg.thumbnail(newW*2,newH*2)
#         # partImg_array = np.uint8(partImg)
#
#         # if newH > 1.5 * newW:
#         #     partImg_array = np.rot90(partImg_array, 1)
#
#         # partImg = Image.fromarray(partImg_array).convert("RGB")
#
#         # partImg.save("./debug_im/{}.jpg".format(index))
#
#         # angel_index = angle_handle.predict(partImg_array)
#         #
#         # angel_class = lable_map_dict[angel_index]
#         # # print(angel_class)
#         # rotate_angle = rotae_map_dict[angel_class]
#         #
#         # if rotate_angle != 0:
#         #     partImg_array = np.rot90(partImg_array, rotate_angle // 90)
#
#         # partImg, box = rotate_cut_img(im, box, leftAdjust, rightAdjust)
#         # partImg = Image.fromarray(partImg_array).convert("RGB")
#         # partImg.save("./debug_im/{}.jpg".format(index))
#         partImg.save(r'outputs/vis_invoice/{}.png'.format(index))
#
#         partImg_ = partImg.convert('L')
#         try:
#             # if crnn_vertical_handle is not None and angel_class in ["shudao", "shuzhen"]:
#             #
#             #     simPred = crnn_vertical_handle.predict(partImg_)
#             # else:
#             #     simPred = crnn_handle.predict(partImg_)  ##识别的文本
#             simPred = recognizer(partImg_)
#         except:
#             continue
#
#         if simPred.strip() != u'':
#             # results.append({'cx': box['cx'] * f, 'cy': box['cy'] * f, 'text': simPred, 'w': box['w'] * f, 'h': box['h'] * f,
#             #                 'degree': box['degree']})
#             results.append({'cx': cx * f, 'cy': cy * f, 'text': simPred, 'w': newW * f, 'h': newH * f,
#                             'degree': degree})
#
#     return results
#
# def crop_rect(img, rect ,alph = 0.2):
#     img  = np.asarray(img)
#     # get the parameter of the small rectangle
#     # print("rect!")
#     # print(rect)
#     center, size, angle = rect[0], rect[1], rect[2]
#     min_size  = min(size)
#
#     if(angle>-45):
#         center, size = tuple(map(int, center)), tuple(map(int, size))
#         # angle-=270
#         size  = ( int(size[0] + min_size*alph ) , int(size[1]  +  min_size*alph) )
#         height, width = img.shape[0], img.shape[1]
#         M = cv2.getRotationMatrix2D(center, angle, 1)
#     # size = tuple([int(rect[1][1]), int(rect[1][0])])
#         img_rot = cv2.warpAffine(img, M, (width, height))
#         # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
#         img_crop = cv2.getRectSubPix(img_rot, size, center)
#     else:
#         center=tuple(map(int,center))
#         size = tuple([int(rect[1][1]), int(rect[1][0])])
#         size  = ( int(size[0] + min_size*alph) ,int(size[1]  + min_size*alph) )
#         angle -= 270
#         height, width = img.shape[0], img.shape[1]
#         M = cv2.getRotationMatrix2D(center, angle, 1)
#         img_rot = cv2.warpAffine(img, M, (width, height))
#         # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
#         img_crop = cv2.getRectSubPix(img_rot, size, center)
#     img_crop = Image.fromarray(img_crop)
#     return img_crop
import os
import subprocess
import cv2
import numpy as np
from PIL import Image
from config import *
from crnn.keys import alphabetChinese,alphabetEnglish
from apphelper.image import rotate_cut_img,sort_box,union_rbox

if ocrFlag=='keras':
    from crnn.network_keras import CRNN
    if chineseModel:
        alphabet = alphabetChinese
        if LSTMFLAG:
            ocrModel = ocrModelKerasLstm
        else:
            ocrModel = ocrModelKerasDense
    else:
        ocrModel = ocrModelKerasEng
        alphabet = alphabetEnglish
        LSTMFLAG = True
elif ocrFlag=='torch':
    from crnn.network_torch import CRNN
    if chineseModel:
        alphabet = alphabetChinese
        if LSTMFLAG:
            ocrModel = ocrModelTorchLstm
        else:
            ocrModel = ocrModelTorchDense

    else:
        ocrModel = ocrModelTorchEng
        alphabet = alphabetEnglish
        LSTMFLAG = True
elif ocrFlag=='opencv':
    from crnn.network_dnn import CRNN
    ocrModel = ocrModelOpencv
    alphabet = alphabetChinese
else:
    print( "err,ocr engine in keras\opencv\darknet")

nclass = len(alphabet)+1
if ocrFlag=='opencv':
    crnn = CRNN(alphabet=alphabet)
else:
    crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
if os.path.exists(ocrModel):
    crnn.load_weights(ocrModel)
else:
    print("download model or tranform model with tools!")

recognizer = crnn.predict

def crnnRec(im, boxes, leftAdjust=False, rightAdjust=False, alph=0.2, f=1.0):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box
    @@ifIm:是否输出box对应的img

    """
    results = []
    # print("orcModel:",ocrModel)
    # print("ocrFlag:",ocrFlag)
    im = Image.fromarray(im)
    # print("boxes:",boxes)
    boxes = sort_box(boxes) # 没有排序
    # print("boxes:",boxes)
    import time
    i=1
    for index, box in enumerate(boxes):
        start_time = time.time()
        degree, w, h, cx, cy = box

        # partImg, newW, newH = rotate_cut_img(im,  90  + degree  , cx, cy, w, h, leftAdjust, rightAdjust, alph)
        partImg = crop_rect(im, ((cx, cy), (h, w), degree))
        newW, newH = partImg.size
        # partImg.thumbnail(newW*2,newH*2)
        # partImg_array = np.uint8(partImg)

        # if newH > 1.5 * newW:
        #     partImg_array = np.rot90(partImg_array, 1)

        # partImg = Image.fromarray(partImg_array).convert("RGB")

        # partImg.save("./debug_im/{}.jpg".format(index))

        # angel_index = angle_handle.predict(partImg_array)
        #
        # angel_class = lable_map_dict[angel_index]
        # # print(angel_class)
        # rotate_angle = rotae_map_dict[angel_class]
        #
        # if rotate_angle != 0:
        #     partImg_array = np.rot90(partImg_array, rotate_angle // 90)

        # partImg, box = rotate_cut_img(im, box, leftAdjust, rightAdjust)
        # partImg = Image.fromarray(partImg_array).convert("RGB")
        # partImg.save("./debug_im/{}.jpg".format(index))


        # partImg.save(r'outputs/vis_invoice/{}.png'.format(index))

        partImg_ = partImg.convert('L')
        try:
            # if crnn_vertical_handle is not None and angel_class in ["shudao", "shuzhen"]:
            #
            #     simPred = crnn_verticv2.cvtColor(cal_handle.predict(partImg_)
            # else:
            pre = time.time()
            simPred = crnn.predict(partImg_)
            # print("simPred:",simPred)##识别的文本
            # simPred = recognizer(partImg_)
        except:
            continue

        if simPred[0].strip() != []:
            # results.append({'cx': box['cx'] * f, 'cy': box['cy'] * f, 'text': simPred, 'w': box['w'] * f, 'h': box['h'] * f,
            #                 'degree': box['degree']})
            # f 默认为 1 
            results.append({'cx': cx * f, 'cy': cy * f, 'text': simPred[0], 'candidate': simPred[1], 'w': newW * f, 'h': newH * f,
                            'degree': degree})
        # print("results:",results)
        i += 1
    return results

def crop_rect(img, rect ,alph = 0.2):
    """

    :param img:
    :param rect: list 里面是元祖形式
    :param alph:
    :return:
    """
    img  = np.asarray(img)
    # get the parameter of the small rectangle
    # print("rect!")
    # print(rect)
    center, size, angle = rect[0], rect[1], rect[2]
    min_size  = min(size)

    if(angle>-45):
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # angle-=270
        size  = ( int(size[0] + min_size*alph ) , int(size[1]  +  min_size*alph) )
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
    # size = tuple([int(rect[1][1]), int(rect[1][0])])
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center=tuple(map(int,center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        size  = ( int(size[0] + min_size*alph) ,int(size[1]  + min_size*alph) )
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("debug_im/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    img_crop = Image.fromarray(img_crop)
    return img_crop