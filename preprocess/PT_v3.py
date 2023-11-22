import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from scipy import signal
import os
import tqdm


def svd(M):
    """
    Args:
        M: numpy matrix of shape (m, n)
    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u,s,v = np.linalg.svd(M, full_matrices=True)

    return u, s, v


def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


def dacm(dt:np.ndarray)->np.ndarray:
    lent=dt.shape[0]
    ret= []
    img=np.imag(dt).tolist()
    rel=np.real(dt).tolist()
    ans = 0
    for j in range(1, lent):
        ans+=((rel[j]*(img[j]-img[j-1])-img[j]*(rel[j]-rel[j-1]))/(rel[j]**2+img[j]**2))
        ret.append(ans)
    return ret


def phase(file_name, file_num, output_pos, adcData, num_ADCSamples = 128, num_chirps = 255, num_frame = 32, num_RX = 8):
    adcData_T0 = adcData[0: num_RX: 1, :] # 8个天线的数据 复数形式

    data = np.zeros((1, num_chirps * num_ADCSamples * num_frame), dtype=complex) # 格式为一个天线的总数据大小
    
    data += adcData_T0[0, :] # 1号天线
    
    # for i in range(4): # 相干积累
        # data += adcData_T0[2 * i, :]
    
    data_frame = np.zeros((num_chirps, num_ADCSamples, num_frame), dtype=complex) # 三维数据
    data_frame_svd = np.zeros_like(data_frame)
    for k in tqdm.tqdm(range(num_frame)):
        for m in range(num_chirps):
            data_frame[m, :, k] = data[0, num_chirps * num_ADCSamples * k + m * num_ADCSamples
                                        : num_chirps * num_ADCSamples * k + (m + 1) * num_ADCSamples]
            U, S, V = svd(data_frame[:, :, k])
            S[0] = 0 # 去除最大值: 静态杂波
            MS = np.zeros_like(data_frame[:, :, k], dtype=complex)
            for i in range(min(num_chirps, num_ADCSamples)):
                MS[i, i] = S[i]
            data_frame_svd[:, :, k] = U @ MS @ V

    # TH_cfar = np.zeros((num_chirps, num_ADCSamples, num_frame), dtype=complex)
    # angle
    max_fft=0
    max_range = 0
    data_fft = np.zeros((num_chirps, num_ADCSamples, num_frame),dtype=complex)
    for k in range(num_frame):
        for m in range(num_chirps):
            data_fft[m,:,k] = np.fft.fft(data_frame_svd[m,:,k])
            data_index = np.argmax(abs(data_fft[m,:,k]))
            if abs(max(data_fft[m,:,k]))>max_fft:
                max_fft = abs(max(data_fft[m,:,k]))
                max_index=data_index
            if abs(max_fft)*0.5>abs(max(data_fft[m,:,k])):
                max_range=max(max_range,abs(max(data_fft[m,:,k])))
    num = eval("1e-"+str(4+int(np.real(np.max(data_fft))//300)))
    num = 1e-5
    # print(max_range)
    dataTest = []
    for k in range(num_frame):
        for m in range(num_chirps):
            data_fft[m,:,k] = np.fft.fft(data_frame_svd[m,:,k])
            # plt.plot(range(num_ADCSamples),data_frame_svd[9,:,k])
            # break
            #print(np.real(max(data_fft[m,:,k])))
            # TH_cfar[m, :, k] = cfar(abs(data_fft[m, :, k]),num)
            # data_fft[m,:,k] = (TH_cfar[m,:,k] < abs(data_fft[m,:,k])) * (data_fft[m,:,k])
            #else:
                #plt.plot(range(num_ADCSamples),data_fft[m,:,k])
            data_index = np.argmax(abs(data_fft[m,:,k]))
            dataTest.append(data_fft[m,data_index,k])
    
    plt.figure()
    plt.plot(range(len(dataTest)-1),dacm(np.array(dataTest)))
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(output_pos+file_name+'_newPT3_'+str(file_num)+'.jpg', pad_inches=0)
    plt.close()
    
    
if __name__ == '__main__':
    file_name = 'data_ooo_smooth_1_rx4_196adc_64f_26_Raw_0'
    output_pos = 'output/new/'
    adcData = io.loadmat('ripe_data/'+file_name+'.mat')
    adcData = adcData['adcData']
    phase(
        file_num = 1,
        file_name = file_name,
        adcData = adcData,
        output_pos = output_pos,
        num_ADCSamples = 196,
        num_chirps = 255,
        num_frame = 64,
    )