from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from skimage.util import random_noise
import os
import numpy as np
import matplotlib.pyplot as plt

#读取图片
def read_PIL(filename):
    img = Image.open("../output/figs/"+filename)
    return img

#旋转
def Rotate(filename,savepath):
    img = read_PIL(filename)
    new_img = img.rotate(2)
    new_img.save(savepath+"Rotate_"+filename)

#加噪
def Noise(filename,savepath):
    img = read_PIL(filename)
    img_arr = np.array(img)
    new_img = random_noise(img_arr, mode='gaussian', mean = 0, var = 0.002)
    #再把numpy数组转回PIL图像
    plt.figure(figsize=(img.size[0], img.size[1]), dpi=1)
    plt.imshow(new_img)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(savepath+"Noise_"+filename)

#调整亮度
def Brightness(filename,savepath):
    img = read_PIL(filename)
    enh_bri = ImageEnhance.Brightness(img)
    new_img1 = enh_bri.enhance(factor=1.5)
    new_img = enh_bri.enhance(factor=0.5)
    new_img.save(savepath+"brightness_"+filename)
    new_img1.save(savepath+"brightness_1_"+filename)

#饱和度
def ColorBalance(filename,savepath):
    img = read_PIL(filename)
    enh_col = ImageEnhance.Color(img)
    new_img = enh_col.enhance(factor=1.5)
    new_img1 = enh_col.enhance(factor=2)
    new_img.save(savepath+"ColorBalabce_"+filename)
    new_img1.save(savepath+"ColorBalabce_1_"+filename)

#对比度
def Contrast(filename,savepath):
    img = read_PIL(filename)
    enh_con = ImageEnhance.Contrast(img)
    new_img = enh_con.enhance(factor=1.5)
    new_img1 = enh_con.enhance(factor=2)
    new_img.save(savepath+"Contrast_"+filename)
    new_img1.save(savepath+"Contrast_1_"+filename)

#锐化
def Sharpness(filename,savepath):
    img = read_PIL(filename)
    enh_sha = ImageEnhance.Sharpness(img)
    new_img = enh_sha.enhance(factor=1.5)
    new_img.save(savepath+"Sharpness_"+filename)


#模糊滤波
def BLUR(filename,savepath):
    img = read_PIL(filename)
    new_img = img.filter(ImageFilter.BLUR)
    new_img.save(savepath+"BLUR_"+filename)

#细节滤波
def DETAIL(filename,savepath):
    img = read_PIL(filename)
    new_img = img.filter(ImageFilter.DETAIL)
    new_img.save(savepath+"DETAIL_"+filename)

#边界增强滤波
def EDGE_ENHANCE(filename,savepath):
    img = read_PIL(filename)
    new_img = img.filter(ImageFilter.EDGE_ENHANCE)
    new_img.save(savepath+"EDGR_ENHANCE_"+filename)

#平滑滤波
def SMOOTH(filename,savepath):
    img = read_PIL(filename)
    new_img = img.filter(ImageFilter.SMOOTH)
    new_img.save(savepath+"SMOOTH_"+filename)

#模式滤波
def ModelFilter(filename,savepath):
    img = read_PIL(filename)
    new_img = img.filter(ImageFilter.ModeFilter(5))
    new_img.save(savepath+"ModelFilter_"+filename)

filenames=os.listdir("../output/figs/")
for filename in filenames:
    Rotate(filename,"../output/figs/")
    Noise(filename,"../output/figs/")
    Brightness(filename,"../output/figs/")
    ColorBalance(filename,"../output/figs/")
    Contrast(filename,"../output/figs/")
    Sharpness(filename,"../output/figs/")
    BLUR(filename,"../output/figs/")
    DETAIL(filename,"../output/figs/")
    EDGE_ENHANCE(filename,"../output/figs/")
    SMOOTH(filename,"../output/figs/")
    ModelFilter(filename,"../output/figs/")