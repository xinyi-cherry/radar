import numpy as np
from cfar import *
import matplotlib.pyplot as plt
import scipy.io as io
from tqdm import tqdm

# 虚拟帧编码 使用单一天线数据(1号天线) 速度维cfar
def cal_RD(
    file_num: int, 
    file_name: str, 
    output_addr: str, 
    adcData: np.ndarray, 
    num_ADCSamples: int = 196, 
    num_chirps: int = 255, 
    num_frames: int = 200,
    num_v_interval: int = 4,         # 虚拟帧chirp间隔
    num_v_chirps: int = 256,         # 虚拟帧chirp数
    num_extract_interval: int = 128, # 抽帧间隔
    pad_width: int = 256,            # 每帧慢时间维度补零个数
    Pfa: float = 1e-10,              # cfar虚警概率
    n_train: int = 10,               # cfar一侧的参考单元长度
    n_guard: int = 5                 # cfar一侧的保护单元的长度
    ):
    
    """
    Using virtual frames to extract chirps.
    Using data of antenna No.1.
    Hamming window.
    Using cfar algorithm in speed dimension.
    """
    # ---------------------------------------原始数据提取---------------------------------------
    adcData_T0 = adcData[0: 4: 1, :] # 8个天线的数据 复数形式
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
    window = np.hamming(pad_chirps)
    data_fft = np.zeros((pad_chirps, num_ADCSamples, num_v_frames), dtype=complex)
    for k in range(num_v_frames):
        temp_frame = data_virtual_frame[:, :, k] * window[:, np.newaxis]
        data_fft[:,:,k] = np.fft.fftshift(np.fft.fft2(temp_frame))
    
    # -----------------------------------------CFAR-----------------------------------------
    #--------------可调参数--------------
    num_V = int(pad_chirps/4) # VT图纵轴长度
    len_R_window_side = 1 # 每帧最大值列两边一侧的保留长度
    len_R_window_total = int(2 * len_R_window_side + 1)
    #-----------------------------------
    TH_cfar = np.zeros((pad_chirps, len_R_window_total, num_v_frames))
    data_temp = np.zeros((pad_chirps, len_R_window_total, num_v_frames))
    data_RD = np.zeros((num_V, num_v_frames))
    VT = np.zeros((num_V, num_v_frames))
    
    energy = np.zeros(num_v_frames)
    flag = 0
    TH_energy = 2e9 # 经验测试值 会随不同抽取方式变化
    TH_last_len = int(num_v_frames/5) # 经验测试值
    signal_tail = num_v_frames - 1
    
    with tqdm(total = num_v_frames) as pbar:
        for k in range(num_v_frames):
            x0, y0 = np.where(abs(data_fft[:, :, k]) == np.max(np.max(abs(data_fft[:, :, k]))))
            data_temp[:, :, k] = abs(data_fft[:, y0[0] - len_R_window_side : y0[0] + len_R_window_side + 1, k])
            for t in range(len_R_window_total):
                TH_cfar[:, t, k] = cfar(signal=abs(data_temp[:, t, k]), Pfa=Pfa, n_train=n_train, n_guard=n_guard)
            data_cfar = (TH_cfar < data_temp) * data_temp
            data_RD[:, k] = np.sum(axis=1, a=(data_cfar[int(pad_chirps/2 - num_V/2): int(pad_chirps/2 + num_V/2), :, k]))
            
            # 静音检测
            energy[k] = np.sum(data_RD[:, k] ** 2)
            if(energy[k] <= TH_energy):
                flag += 1
                if(flag == TH_last_len):
                    signal_tail = k - int(TH_last_len / 2)
                    tqdm.write("Exception: Find the end frame.")
                    pbar.n = pbar.last_print_n = num_v_frames
                    pbar.update(0)
                    break
            else:
                flag = 0
            
            tmp = data_RD[:, k] / (np.max(data_RD[:, k]) + np.finfo(float).eps)
            VT[:, k] = tmp
            pbar.update(1)
    
    VT[:, signal_tail:] = 0
    
    # ----------------------------------------输出图像----------------------------------------
    plt.figure(figsize=(num_v_frames, num_V), dpi=1)
    plt.imshow(abs(VT), cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(
        f'{output_addr}{file_name}_vi_vc_ei_{num_v_interval}_{num_v_chirps}_{num_extract_interval}_VT8_{file_num}.jpg',
        transparent=True,
        dpi=1,
        pad_inches=0
    )
    print(f'{file_name} VT8 complete')


if __name__ == '__main__':
    file_name = 'data_cth_3_1_Raw_0'
    output_addr = 'output/new/'
    adcData = io.loadmat(f'ripe_data/new/{file_name}.mat')
    adcData = adcData['adcData']
    
    cal_RD(
        file_num = 1,
        file_name = file_name,
        output_addr = output_addr,
        adcData = adcData,
        num_ADCSamples = 196,
        num_chirps = 255,
        num_frames = 200,
        
        num_v_interval          = 4,    # 4     # 虚拟帧chirp间隔
        num_v_chirps            = 256,  # 256   # 虚拟帧chirp数
        num_extract_interval    = 128,  # 128   # 抽帧间隔
        pad_width               = 256,  # 256   # 每帧慢时间维度补零个数
        
        Pfa         = 1e-3, # 1e-3  # cfar虚警概率 
        n_train     = 15,   # 15    # cfar一侧的参考单元长度
        n_guard     = 20    # 20    # cfar一侧的保护单元的长度
    )