import os
from shutil import copy
import random
from PIL import Image
import cv2
import numpy as np

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'E:\BaiduNetdiskDownload\love'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]

mkfile('E:\BaiduNetdiskDownload\love/train')
for cla in flower_class:
    mkfile('E:\BaiduNetdiskDownload\love/train/'+cla)

mkfile('E:\BaiduNetdiskDownload\love/val')
for cla in flower_class:
    mkfile('E:\BaiduNetdiskDownload\love/val/'+cla)

split_rate = 0.1
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'E:\BaiduNetdiskDownload\love/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'E:\BaiduNetdiskDownload\love/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

print("processing done!")


#将单通道图像转化为3通道图像
file = r'D:\image'
floder = [cla for cla in os.listdir(file)]
total = 0
flodernumber = 0
for cla in floder:
    flodernumber+=1
    print(cla,'   正在处理第',flodernumber,'个文件夹')
    images = os.listdir(os.path.join(file,cla))
    for image_path in images:
        image_path = os.path.join(file,cla,image_path)
        img = Image.open(image_path)
        if len(img.split()) != 3:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            cv2.imwrite(image_path, img)
            total += 1
            print("已处理一张图片")

print('共处理图片数量： ',total)
