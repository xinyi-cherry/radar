import numpy as np
from cfar import *
import matplotlib.pyplot as plt
import scipy.io as io
import tqdm

# 虚拟帧编码 使用单一天线数据(1号天线) 图像进行杂波滤除 图像大小
def cal_RD(
    file_num, 
    file_name, 
    output_pos, 
    adcData, 
    num_ADCSamples = 128, 
    num_chirps = 255, 
    num_frames = 32,
    num_v_interval = 2, # 虚拟帧chirp间隔    # 2
    num_v_chirps = 160, # 虚拟帧chirp数      # 160
    num_extract_interval = 80, # 抽帧间隔    # 81
    pad_width = 200, # 每帧慢时间维度补零个数 # 200
    Pfa = 1e-10, # cfar虚警概率 # 1e-10
    n_train = 10, # cfar一侧的参考单元长度 # 10
    n_guard = 5, # cfar一侧的保护单元的长度 # 5
    ):

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
    # num_v_interval = 2 # 虚拟帧chirp间隔    # 2
    # num_v_chirps = 160 # 虚拟帧chirp数      # 160
    # num_extract_interval = 80 # 抽帧间隔    # 81
    # pad_width = 200 # 每帧慢时间维度补零个数 # 200
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
        # plt.figure()
        # plt.pcolormesh(np.arange(num_ADCSamples),np.arange(pad_chirps), abs(data_fft[:,:,k]), shading='auto')
        # plt.xlabel("fast")
        # plt.ylabel("slow")
        # plt.show()
    
    
    # -----------------------------------------CFAR-----------------------------------------
    #--------------可调参数--------------
    num_R = int(num_ADCSamples/2) # RT图纵轴长度
    num_V = int(pad_chirps/4) # VT图纵轴长度
    window_width = 2 # 两侧长度
    
    # Pfa = 1e-10 # cfar虚警概率 # 1e-10
    # n_train = 10 # cfar一侧的参考单元长度 # 10
    # n_guard = 5 # cfar一侧的保护单元的长度 # 5
    # cut_width = 10 # VT保留窗口半宽
    #-----------------------------------
    TH_cfar = np.zeros((pad_chirps, num_ADCSamples, num_v_frames), dtype=complex)
    data_RD = np.zeros((num_V, num_R, num_v_frames))
    VT = np.zeros((num_V, num_v_frames))
    
    for k in tqdm.tqdm(range(num_v_frames)):
        for m in range(pad_chirps):
            TH_cfar[m, :, k] = cfar(signal=abs(data_fft[m, :, k]), Pfa=Pfa, n_train=n_train, n_guard=n_guard)
        data_cfar = (TH_cfar < abs(data_fft)) * abs(data_fft)
        data_RD[:, :, k] = data_cfar[int(pad_chirps/2 - num_V / 2): int(pad_chirps/2 + num_V / 2), int(num_ADCSamples/2): int(num_ADCSamples/2 + num_R), k]
        # plt.figure()
        # plt.pcolormesh(np.arange(num_R),np.arange(num_V), abs(data_RD[:,:,k]), shading='auto')
        # plt.xlabel("fast")
        # plt.ylabel("slow")
        # plt.show()
        x0, y0 = np.where(data_RD[:, :, k] == np.max(np.max(data_RD[:, :, k])))
        tmp = data_RD[:, y0[0], k].T
        tmp = tmp / (np.max(tmp) + np.finfo(float).eps)
        VT[:, k] = tmp
    
    # ----------------------------------------输出图像----------------------------------------
    plt.figure(figsize=(num_v_frames, num_V), dpi=1)
    plt.imshow(abs(VT), cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(output_pos+file_name+'_VT5_'+str(file_num)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    print(file_name + ' VT5 complete')


if __name__ == '__main__':
    file_name = 'data_ababa_4_rx4_196adc_64f_26.12p_Raw_0'
    output_pos = 'output/new/'
    adcData = io.loadmat('ripe_data/'+file_name+'.mat')
    adcData = adcData['adcData']
    
    cal_RD(
        file_num = 1,
        file_name = file_name,
        output_pos = output_pos,
        adcData = adcData,
        num_ADCSamples = 196,
        num_chirps = 255,
        num_frames = 64,
        num_v_interval = 2,             # 2(rx8)  # 2(rx4)
        num_v_chirps = 256,             # 160       256
        num_extract_interval = 128,     # 81        128
        pad_width = 256,                # 200       256
        Pfa = 1e-10, # cfar虚警概率 # 1e-10
        n_train = 10, # cfar一侧的参考单元长度 # 10
        n_guard = 5, # cfar一侧的保护单元的长度 # 5
    )