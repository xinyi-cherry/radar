import numpy as np
import scipy.io as sio
import os
def bin2matNew(filename:str = 'slope6.5_ababa_Raw_1',ORIGINrootPosition:str="./raw_data/",AFTERrootPosition:str='./ripe_data/',
                num_ADC = 4096, num_chirps_pframe = 64, num_RX = 8):
    # filename = 'slope6.5_ababa_Raw_1'

    fid = open(ORIGINrootPosition+filename+".bin", 'rb')
    adcData = np.fromfile(fid, dtype=np.int16)
    fid.close()

    # num_ADC = 4096
    # num_chirps_pframe = 1
    # num_RX = 8
    filesize = adcData.size
    num_frames = filesize // (2 * num_ADC * num_chirps_pframe * num_RX) # 2:IQ

    print("\033[34m %s@%s::\033[0m" % (filename,os.getpid())+'filesize = ', filesize)
    print("\033[34m %s@%s::\033[0m" % (filename,os.getpid())+'num_frames = ', num_frames)

    data_frame = np.zeros((num_RX, filesize // (2 * num_RX)), dtype=complex) # 2:IQ

    step_frame_in = num_ADC * num_chirps_pframe * 2 * num_RX # 2:IQ
    step_chirp_in = num_ADC * 2 * num_RX # 2:IQ
    step_RX_in = num_ADC * 2 # 2:IQ
    step_frame_out = num_ADC * num_chirps_pframe # 128 * 160

    for frame in range(num_frames):
        for chirp in range(num_chirps_pframe):
            for RX_i, RX in enumerate([0,2,4,6,1,3,5,7]):
                start_in = int(frame * step_frame_in + chirp * step_chirp_in + RX_i * step_RX_in)
                end_in = int(frame * step_frame_in + chirp * step_chirp_in + (RX_i+1) * step_RX_in)
                start_out = int(frame * step_frame_out + chirp * num_ADC)
                end_out = int(frame * step_frame_out + (chirp+1) * num_ADC)
                # 奇数数据
                data_frame[RX, start_out:end_out:2] = adcData[start_in:end_in:4] + 1j*adcData[start_in+2:end_in:4]
                # 偶数数据
                data_frame[RX, start_out+1:end_out+1:2] = adcData[start_in+1:end_in:4] + 1j*adcData[start_in+3:end_in:4]

    adc_Data = {'adcData':data_frame}
    sio.savemat(AFTERrootPosition+filename+'_test.mat', adc_Data)
    print("\033[34m %s@%s::\033[0m" % (filename,os.getpid())+'Converted data saved in:'+AFTERrootPosition+filename+'_test.mat')