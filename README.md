# OCR项目(发票识别，驾驶证识别，身份证识别，营业执照识别)

## Requirements
* Python 3.6
* PyTorch 1.8
* pyclipper
* Polygon2
* opencv-python 4.0.0.21



##几个必要的文件说明
```
.
├── app                          执行 python app.py 时，从这个目录下调，
│   ├── businesslicense_gao.py   仿写的
│   └── businesslicense.py       原版
│    ...                         其他类似
├── app.py                       执行 python app.py ,从浏览器访问 localhost/#/test，用于向用户展示识别效果
├── crnn                         文本识别模型
│   ├── ...
│    ...
│   └── ...
├── dbnet                        文本检测模型
│   ├── ...
│    ...
│   └── ...
├── test_tmp.py                  执行 python test_tmp.py 用于测试检测和识别效果
└── test_app.py                  执行 python app.py 会被调用

```
## 模型及数据集

```
crnn文本识别模型放在./model/recognition/下
dbnet文本识别模型根据 init_args()函数里的 --model_path 存放
模型链接：https://pan.baidu.com/s/1_rSLbprQCTfhj0819FbFAA 提取码:8ma7
数据集链接：https://pan.baidu.com/s/1TOVmWaLmilqGszr8Pw7Kxw 提取码: g42d
```