import numpy as np
from cfar import *
import matplotlib.pyplot as plt
import scipy.io as io
import os

# 虚拟帧编码 使用单一天线数据(1号天线) 图像进行杂波滤除 图像大小
def cal_RD(file_num, file_name, adcData, num_ADCSamples = 128, num_chirps = 255, num_frames = 96):
    # ---------------------------------------原始数据提取---------------------------------------
    adcData_T0 = adcData[0: 8: 1, :] # 8个天线的数据 复数形式
    print(adcData.shape)
    data = np.zeros((1, num_chirps * num_ADCSamples * num_frames), dtype=complex) # 格式为一个天线的总数据大小
    data += adcData_T0[0, :] # 1号天线
    data_frame = np.zeros((num_chirps, num_ADCSamples, num_frames), dtype=complex) # 三维数据
    for k in range(num_frames):
        for m in range(num_chirps):
            data_frame[m, :, k] = data[0, num_chirps * num_ADCSamples * k + m * num_ADCSamples
                                        : num_chirps * num_ADCSamples * k + (m + 1) * num_ADCSamples]

    # ----------------------------------------拉直处理----------------------------------------
    num_all_chirps = num_frames * num_chirps
    data_spread = np.zeros((num_all_chirps, num_ADCSamples), dtype=complex)
    for k in range(num_frames):
        for m in range(num_chirps): # 两天线数据交叉合并
            data_spread[num_chirps * k + m, :] = data_frame[m, :, k]

    # ------------------------------------------MTI------------------------------------------
    #--------------可调参数--------------
    mti_width = 2
    #-----------------------------------
    num_mti_all = num_all_chirps - mti_width
    data_mti_spread = np.zeros((num_mti_all, num_ADCSamples), dtype=complex)
    for m in range(num_mti_all):
        data_mti_spread[m, :] = data_spread[m+mti_width, :] - data_spread[m, :]

    # ---------------------------------------虚拟帧编码---------------------------------------
    #--------------可调参数--------------
    num_v_interval = 2 # 虚拟帧chirp间隔
    num_v_chirps = 160 # 虚拟帧chirp数
    num_extract_interval = 81 # 抽帧间隔
    pad_width = 200 # 每帧慢时间维度补零个数
    #-----------------------------------
    num_frame_real_len = num_v_interval * (num_v_chirps-1) + 1
    num_v_frames = int(((num_mti_all - num_frame_real_len) // num_extract_interval) + 1)
    pad_chirps = num_v_chirps + pad_width
    print('real length of a virtual frame: ', num_frame_real_len)
    print('number of virtual frames: ', num_v_frames)
    data_virtual_frame = np.zeros((pad_chirps, num_ADCSamples, num_v_frames), dtype=complex)
    for k in range(num_v_frames):
        temp = np.pad(data_mti_spread[(k*num_extract_interval): (k*num_extract_interval + num_frame_real_len): num_v_interval, :], ((0,pad_width),(0,0)), mode='constant', constant_values=0+0j)
        data_virtual_frame[:, :, k] = temp

    # ----------------------------------------二维FFT----------------------------------------
    data_fft = np.zeros((pad_chirps, num_ADCSamples, num_v_frames), dtype=complex)
    for k in range(num_v_frames):
        data_fft[:,:,k] = np.fft.fftshift(np.fft.fft2(data_virtual_frame[:, :, k]))

    # -----------------------------------------CFAR-----------------------------------------
    #--------------可调参数--------------
    num_R = int(num_ADCSamples/2) # RT图纵轴长度
    num_V = int(pad_chirps/6) # VT图纵轴长度
    cut_width = 10 # VT保留窗口半宽
    #-----------------------------------
    TH_cfar = np.zeros((pad_chirps, num_ADCSamples, num_v_frames), dtype=complex)
    data_RD = np.zeros((num_V, num_R, num_v_frames))
    VT = np.zeros((num_V, num_v_frames))
    for k in range(num_v_frames):
        for m in range(pad_chirps):
            TH_cfar[m, :, k] = cfar(abs(data_fft[m, :, k]))
        data_cfar = (TH_cfar < abs(data_fft)) * abs(data_fft)
        data_RD[:, :, k] = data_cfar[int(pad_chirps/2 - num_V / 2): int(pad_chirps/2 + num_V / 2), int(num_ADCSamples/2): int(num_ADCSamples/2 + num_R), k]
        x0, y0 = np.where(data_RD[:, :, k] == np.max(np.max(data_RD[:, :, k])))
        tmp = data_RD[:, y0[0], k].T
        tmp = tmp / (np.max(tmp) + np.finfo(float).eps)
        VT[:, k] = tmp
        if(k % 10 ==0):
            print("\033[34m %s@%s::\033[0m" % (file_name,os.getpid())+'Frame %d complete.' % (k))
    
    # 杂波滤除
    new_VT = np.zeros((num_V, num_v_frames))
    for k in range(num_v_frames):
        max_y = np.argmax(VT[:,k])
        new_VT[:,k] = VT[:,k]
        mask = np.zeros_like(new_VT[:,k], dtype=bool) # 布尔掩码 用于将选取索引外赋值为0
        mask[max_y-cut_width:max_y+cut_width] = True
        new_VT[:,k] = new_VT[:,k] * mask
    
    # ----------------------------------------输出图像----------------------------------------
    plt.figure(figsize=(num_v_frames, num_V), dpi=1)
    plt.imshow(abs(new_VT), cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('output/figs/'+file_name+'_VT5'+str(file_num)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    print("\033[34m %s@%s::\033[0m" % (file_name,os.getpid()) + file_name + ' new_VT complete')
    plt.close()
    

if __name__ == '__main__':
    file_name = 'cth_10_11_Raw_0'
    adcData = io.loadmat('ripe_data/'+file_name+'.mat')
    adcData = adcData['adcData']
    cal_RD(file_num=1, file_name=file_name, adcData=adcData)