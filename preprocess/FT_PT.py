import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from scipy import signal
import os

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

def phase(type,file_position, file_num, file_name, adcData, num_ADCSamples = 128, num_chirps = 255, num_frame = 96):
    adcData_T0 = adcData[0: 8: 1, :] # 8个天线的数据 复数形式

    data = np.zeros((1, num_chirps * num_ADCSamples * num_frame), dtype=complex) # 格式为一个天线的总数据大小
    
    data += adcData_T0[2, :] # 3号天线
    
    # for i in range(4): # 相干积累
        # data += adcData_T0[2 * i, :]
    
    data_frame = np.zeros((num_chirps, num_ADCSamples, num_frame), dtype=complex) # 三维数据
    for k in range(num_frame):
        for m in range(num_chirps):
            data_frame[m, :, k] = data[0, num_chirps * num_ADCSamples * k + m * num_ADCSamples
                                        : num_chirps * num_ADCSamples * k + (m + 1) * num_ADCSamples]

    # MTI
    num_mti = num_chirps - 2 # 62
    data_mti = np.zeros((num_mti, num_ADCSamples, num_frame), dtype=complex)
    for k in range(num_frame):
        for m in range(num_mti):
            data_mti[m,:,k] = data_frame[m+2,:,k] - data_frame[m,:,k]
    
    TH_cfar = np.zeros((num_mti, num_ADCSamples, num_frame), dtype=complex)
    # angle
    max_index=0
    max_fft=0
    max_range = 0
    angle_fft = np.zeros((num_mti,num_frame),dtype=float)
    data_fft = np.zeros((num_mti, num_ADCSamples, num_frame),dtype=complex)
    chirp_average = 0
    #plt.figure(3)
    for k in range(num_frame):
        for m in range(num_mti):
            data_fft[m,:,k] = np.fft.fft(data_mti[m,:,k])
            data_index = np.argmax(abs(data_fft[m,:,k]))
            if abs(max(data_fft[m,:,k]))>max_fft:
                max_fft = abs(max(data_fft[m,:,k]))
                max_index=data_index
            if abs(max_fft)*0.5>abs(max(data_fft[m,:,k])):
                max_range=max(max_range,abs(max(data_fft[m,:,k])))
    num = eval("1e-"+str(4+int(np.real(np.max(data_fft))//300)))
    num = 1e-5
    print(max_range)
    for k in range(num_frame):
        for m in range(num_mti):
            data_fft[m,:,k] = np.fft.fft(data_mti[m,:,k])
            # plt.plot(range(num_ADCSamples),data_mti[9,:,k])
            # break
            #print(np.real(max(data_fft[m,:,k])))
            # TH_cfar[m, :, k] = cfar(abs(data_fft[m, :, k]),num)
            # data_fft[m,:,k] = (TH_cfar[m,:,k] < abs(data_fft[m,:,k])) * (data_fft[m,:,k])
            
            if max(abs(data_fft[m,:,k]))<max_range*1.1:
                if k!=0:
                    angle_fft[m,k] = angle_fft[m,k-1]
                else:
                    angle_fft[m,k] = angle_fft[m-1,-1]
            #else:
                #plt.plot(range(num_ADCSamples),data_fft[m,:,k])
            data_index = np.argmax(abs(data_fft[m,:,k]))
            data_angle = np.angle(data_fft[m,data_index,k])
            angle_fft[m,k] = data_angle
        #break
    #plt.show()
    #plt.close()
    print(file_name)
    stft_data = np.asarray([])
    for k in range(num_frame):
        for m in range(num_mti):
            stft_data=np.append(stft_data,data_mti[m,max_index,k])
    plt.figure()
    #plt.figure(figsize=(10000,4096),dpi=1)
    plt.specgram(stft_data, NFFT=256, Fs=1, noverlap=128,pad_to=1024,detrend='linear')
    plt.ylim((-0.04,0.04))
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(file_position+file_name+'_FT_256_'+str(file_num)+'.jpg',pad_inches=0)
    plt.close()
    plt.figure()
    #plt.figure(figsize=(10000,4096),dpi=1)
    plt.specgram(stft_data, NFFT=512, Fs=1, noverlap=256,pad_to=2048,detrend='linear')
    plt.ylim((-0.04,0.04))
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(file_position+file_name+'_FT_512_'+str(file_num)+'.jpg', pad_inches=0)
    plt.close()
    plt.figure()
    #plt.figure(figsize=(10000,4096),dpi=1)
    plt.specgram(stft_data, NFFT=1024, Fs=1, noverlap=512,pad_to=4096,detrend='linear')
    plt.ylim((-0.04,0.04))
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(file_position+file_name+'_FT_1024_'+str(file_num)+'.jpg', pad_inches=0)
    plt.close()
    plt.figure()
    #plt.figure(figsize=(10000,4096),dpi=1)
    plt.specgram(stft_data, NFFT=768, Fs=1, noverlap=384,pad_to=3072,detrend='linear')
    plt.ylim((-0.04,0.04))
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(file_position+file_name+'_FT_768_'+str(file_num)+'.jpg', pad_inches=0)
    plt.close()
    print("\033[34m %s@%s::\033[0m" % (file_name,os.getpid()) +'saved::'+ file_position+file_name+'_FT_'+'x'+'_1'+'.jpg' + 'complete!')
    # print(num_frame*num_chirps)
    plt.figure()
    angle_data = angle_fft[:,0]
    for k in range(1,num_frame):
        angle_data = np.append(angle_data,angle_fft[:,k])
    #plt.plot(range(len(angle_data)-2),angle_data[:-2])
    #angle_data = np.asarray(double_exponential_smoothing(angle_data,0.03,0.03))
    
    for i in range(1,len(angle_data)):
        diff = angle_data[i] - angle_data[i-1]
        if diff>np.pi:
            angle_data[i:] = angle_data[i:] - 2*np.pi
        elif diff<-np.pi:
            angle_data[i:] = angle_data[i:] + 2*np.pi
    
    angle_data = np.asarray(double_exponential_smoothing(angle_data,0.03,0.03))
    #angle_data = angle_data[::128]
    # for i in range(0,len(angle_data)-1):
    #     angle_data[i] -= angle_data[i+1]
    plt.plot(range(len(angle_data)-2),angle_data[:-2])
    
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.show()
    plt.savefig(file_position+file_name+'_PT_'+str(file_num)+'.jpg', pad_inches=0)
    plt.close()
    
if __name__ == '__main__':
    file_name = 'data1_3_Raw_0_test'
    adcData = io.loadmat('ripe_data/'+file_name+'.mat')
    adcData = adcData['adcData']
    phase(file_num=1, file_name=file_name, adcData=adcData,type=1,file_position='./')