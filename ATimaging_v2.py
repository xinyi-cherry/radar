import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
# from scipy import misc
# from cv2 import imresize
import cv2

# adcData = io.loadmat('testData\\1.mat')
# cluster_data = io.loadmat('F:\本科毕设—双频干涉SAR\现有代码\干涉数据\\net_output1.mat')
# adcData = adcData['adcData']


def cal_AT(type, file_num, file_name, adcData, num_ADCSamples = 128, num_chirps = 255, num_frame = 64, num_RX = 8):
    c = 3e8
    f0 = 77e9
    lamda = c / f0
    data = np.zeros((num_RX, num_ADCSamples, num_chirps, num_frame), dtype=complex)
    for nn in range(num_RX):
        index = 0
        for ii in range(num_frame):
            for jj in range(num_chirps):
                data[nn, :, jj, ii] = adcData[nn, (index * num_ADCSamples):(index + 1) * num_ADCSamples]
                index += 1

    interval = 3
    num_MTI = num_chirps - interval  # 双脉冲对消间隔2
    data_MTI = np.zeros((num_RX, num_ADCSamples, num_MTI, num_frame), dtype=complex)
    for nn in range(num_RX):
        for ii in range(num_frame):
            for jj in range(num_MTI):
                data_MTI[nn, :, jj, ii] = data[nn, :, jj, ii] - data[nn, :, jj + interval, ii]

    d_base = 0.5 * lamda  # 基线长度

    # 方位角
    d = np.arange(0, 3 * d_base, d_base)
    space_num = 101
    angle = np.linspace(-50, 50, space_num)  # 用于存放幅度-角度曲线横轴
    Pmusic1 = np.zeros(space_num)  # 用于存放幅度-角度曲线
    Pmusic2 = np.zeros(space_num)  # 用于存放幅度-角度曲线
    Pmusic_mn = np.zeros((space_num, num_frame * num_MTI))  # 用于存放AT图

    index = 0  # 脉冲计数
    for ii in range(num_frame):  # 遍历32帧，分别求取每帧中间一个PRT的AT图
        for jj in range(num_MTI):
            Rxx = data_MTI[2:7:2, :, jj, ii] @ data_MTI[2:7:2, :, jj, ii].conj().T / num_ADCSamples  # 特征值分解
            K = Rxx.shape[0]
            J = np.flip(np.eye(K),axis=0)
            Rxx = Rxx + J@Rxx.conj()@J

            D, EV = np.linalg.eig(Rxx)  # 特征值分解
            I = D.argsort()  # 将特征值排序从小到大
            EV = np.fliplr(EV[:, I])  # 对应特征矢量排序

            # 遍历每个角度，计算空间谱
            for iang in range(space_num):
                phim = np.deg2rad(angle[iang])
                a = np.exp(-1j * 2 * np.pi * d / lamda * np.sin(phim))
                En = EV[:, 1:]  # 取矩阵的第M+1到N列组成噪声子空间
                Pmusic1[iang] = 1 / abs(a.conj().T @ (En @ En.conj().T) @ a)

            Rxx = data_MTI[3:8:2, :, jj, ii] @ data_MTI[3:8:2, :, jj, ii].conj().T / num_ADCSamples  # 特征值分解
            K = Rxx.shape[0]
            J = np.flip(np.eye(K),axis=0)
            Rxx = Rxx + J@Rxx.conj()@J
            D, EV = np.linalg.eig(Rxx)  # 特征值分解
            I = D.argsort()  # 将特征值排序从小到大
            EV = np.fliplr(EV[:, I])  # 对应特征矢量排序

            # 遍历每个角度，计算空间谱
            for iang in range(space_num):
                phim = np.deg2rad(angle[iang])
                a = np.exp(-1j * 2 * np.pi * d / lamda * np.sin(phim))
                En = EV[:, 1:]  # 取矩阵的第M+1到N列组成噪声子空间
                Pmusic2[iang] = 1 / abs(a.conj().T @ (En @ En.conj().T) @ a)
            
            index = index + 1
            Pmusic_abs = abs(Pmusic1 + Pmusic2)
            Pmmax = max(Pmusic_abs)
            Pmmax_arg = np.argmax(Pmusic_abs)
            # Pmusic_mn[:, index-1] = ((Pmusic_abs / Pmmax)) # 归一化处理
            Pmusic_mn[:, index-1] = Pmusic_abs # 不归一化处理
            # Pmusic_mn[Pmmax_arg, index-1] = 255 # 保留最大值并赋值255
            
    # AT_FW = misc.imresize(Pmusic_mn, [32, 32])

    # 俯仰角
    d = np.arange(0, 4 * d_base, d_base)
    space_num = 91
    angle = np.linspace(-45, 45, space_num)  # 用于存放幅度-角度曲线横轴
    Pmusic = np.zeros(space_num)  # 用于存放幅度-角度曲线
    Pmusic_mmn = np.zeros((space_num, num_frame * num_MTI))  # 用于存放AT图

    index = 0  # 脉冲计数
    for ii in range(num_frame):  # 遍历32帧，分别求取每帧中间一个PRT的AT图
        for jj in range(num_MTI):
            Rxx = data_MTI[0:4, :, jj, ii] @ data_MTI[0:4, :, jj, ii].conj().T / num_ADCSamples  # 特征值分解
            K = Rxx.shape[0]
            J = np.flip(np.eye(K),axis=0)
            Rxx = Rxx + J@Rxx.conj()@J
            D, EV = np.linalg.eig(Rxx)  # 特征值分解
            I = D.argsort()  # 将特征值排序从小到大
            EV = np.fliplr(EV[:, I])  # 对应特征矢量排序
            
            # 遍历每个角度，计算空间谱
            for iang in range(space_num):
                phim = np.deg2rad(angle[iang])
                a = np.exp(-1j * 2 * np.pi * d / lamda * np.sin(phim))
                En = EV[:, 1:]  # 取矩阵的第M+1到N列组成噪声子空间
                Pmusic[iang] = 1 / abs(a.conj().T @ (En @ En.conj().T) @ a)

            index = index + 1
            Pmusic_abs = abs(Pmusic)
            Pmmax = max(Pmusic_abs)
            Pmmax_arg = np.argmax(Pmusic_abs)
            # Pmusic_mmn[:, index-1] = ((Pmusic_abs / Pmmax)) # 归一化处理
            Pmusic_mmn[:, index-1] = Pmusic_abs # 不归一化处理
            # Pmusic_mmn[Pmmax_arg, index-1] = 255 # 保留最大值并二值化

    fig, axs = plt.subplots(1, 2)
    fig_x = 32
    fig_y = Pmusic_mn.shape[0]
    
    axs[0].imshow(abs(Pmusic_mn), aspect='auto', cmap='gray')
    axs[0].set_xlabel('Frames')
    axs[0].set_ylabel('Angle')
    axs[0].set_title('AT_FW')

    axs[1].imshow(abs(Pmusic_mmn), aspect='auto', cmap='gray')
    axs[1].set_xlabel('Frames')
    axs[1].set_ylabel('Angle')
    axs[1].set_title('AT_FY')

    AT_FW = cv2.resize(abs(Pmusic_mn), (fig_x, fig_y))
    plt.figure(figsize=(fig_x, fig_y), dpi=1)
    plt.imshow(AT_FW, cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    plt.savefig('output/'+file_name+'_AT_FW'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    print(file_name + ' FW complete')
    
    # if type == 'B':
    #     plt.savefig('/home/ubuntu/my_codes/BB_val/AT_FW/AT_FW_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # elif type == 'F':
    #     plt.savefig('/home/ubuntu/my_codes/FF_val/AT_FW/AT_FW_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # elif type == 'L':
    #     plt.savefig('/home/ubuntu/my_codes/LL_val/AT_FW/AT_FW_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # elif type == 'R':
    #     plt.savefig('/home/ubuntu/my_codes/RR_val/AT_FW/AT_FW_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)

    AT_FY = cv2.resize(abs(Pmusic_mmn), (fig_x, fig_y))
    plt.figure(figsize=(fig_x, fig_y), dpi=1)
    plt.imshow(AT_FY, cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    plt.savefig('output/'+file_name+'_AT_FY'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    print(file_name + ' FY complete')
    
    # plt.savefig('./AT_FY/AT_FY_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # if type == 'B':
    #     plt.savefig('/home/ubuntu/my_codes/BB_val/AT_FY/AT_FY_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # elif type == 'F':
    #     plt.savefig('/home/ubuntu/my_codes/FF_val/AT_FY/AT_FY_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # elif type == 'L':
    #     plt.savefig('/home/ubuntu/my_codes/LL_val/AT_FY/AT_FY_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)
    # elif type == 'R':
    #     plt.savefig('/home/ubuntu/my_codes/RR_val/AT_FY/AT_FY_'+str(file_num+1)+'.jpg', transparent=True, dpi=1, pad_inches=0)