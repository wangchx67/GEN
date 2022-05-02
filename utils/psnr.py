import torch
import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt


def calc_psnr(img1, img2):
    mean=torch.mean((img1 - img2)**2)
    psnr=10. * torch.log10(1. / (mean))
    return psnr

def calc_psnr_rgb(img1, img2):
    mean = np.mean((img1 - img2) ** 2)
    psnr = 10. * np.log10(1. / (mean))
    return psnr

def make_dataset():  # 读取自己的数据的函数

    dataset_list = []
    # dirgt = "/data/fivek/eval/expertC_gt/"
    # dirimg = "/data/L_E_Data/chenxi_fivek_test_output/fivek_enhanced_GLAD/"
    #dirimg = "/home/wangchenxi/projects/color_dark/data/result/fivek/eval/input/"

    dirgt = "/data/fivek/eval/expertC_gt/"
    dirimg = "/home/wangchenxi/projects/color_dark/data/result/fivek/"


    for fGT in glob.glob(os.path.join(dirimg, '*.*')):
        fName = os.path.basename(fGT)
        fName_=fName.replace('_enhanced_img','')
        dataset_list.append([os.path.join(dirimg, fName), os.path.join(dirgt, fName_)])

    return dataset_list

if __name__ == "__main__":
    data_list=make_dataset()
    psnr=0.0
    x = np.arange(500)
    y=[]
    plt.figure()
    for file in data_list:
        #print(file)
        img=Image.open(file[0])
        img = (np.asarray(img)/255.0)
        #img = torch.from_numpy(img).float().permute(2, 0, 1)
        gt=Image.open(file[1])
        gt = (np.asarray(gt)/255.0)
        #gt = torch.from_numpy(gt).float().permute(2, 0, 1)
        psnr_=calc_psnr_rgb(img,gt)
        y.append(psnr_)
        psnr=psnr+psnr_
    print(psnr/500)
    p1 = plt.bar(x, height=y, width=0.5, )
    plt.show()
