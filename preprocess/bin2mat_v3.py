import numpy as np
import scipy.io as io
def bin2mat(file_name, output_pos, num_ADCSamples, num_chirps, num_RX = 8):
    
    fid = open("./raw_data/"+file_name+".bin", 'rb')
    adcData = np.fromfile(fid, dtype=np.int16)
    fid.close()

    filesize = adcData.size
    num_frames = filesize // (2 * num_ADCSamples * num_chirps * num_RX) # 2:IQ

    print('filesize = ', filesize)
    print('num_frames = ', num_frames)

    data_frame = np.zeros((num_RX, filesize // (2 * num_RX)), dtype=complex) # 2:IQ

    step_frame_in = num_ADCSamples * num_chirps * 2 * num_RX # 2:IQ
    step_chirp_in = num_ADCSamples * 2 * num_RX # 2:IQ
    step_RX_in = num_ADCSamples * 2 # 2:IQ
    step_frame_out = num_ADCSamples * num_chirps # 128 * 160
    RX_list = []
    if num_RX == 8:
        RX_list = [0,2,4,6,1,3,5,7]
    elif num_RX == 4:
        RX_list = [0,1,2,3]
    else:
        print("Invalid RX!")
        return
    for frame in range(num_frames):
        for chirp in range(num_chirps):
            for RX_i, RX in enumerate(RX_list):
                start_in = int(frame * step_frame_in + chirp * step_chirp_in + RX_i * step_RX_in)
                end_in = int(frame * step_frame_in + chirp * step_chirp_in + (RX_i+1) * step_RX_in)
                start_out = int(frame * step_frame_out + chirp * num_ADCSamples)
                end_out = int(frame * step_frame_out + (chirp+1) * num_ADCSamples)
                # 奇数数据
                data_frame[RX, start_out:end_out:2] = adcData[start_in:end_in:4] + 1j*adcData[start_in+2:end_in:4]
                # 偶数数据
                data_frame[RX, start_out+1:end_out+1:2] = adcData[start_in+1:end_in:4] + 1j*adcData[start_in+3:end_in:4]

    adc_Data = {'adcData':data_frame}
    io.savemat(output_pos+file_name+'.mat', adc_Data)
    return num_frames
    
    
if __name__ == '__main__':
    file_name = 'data_nearababa_1_rx4_196adc_64f_26p11_Raw_0'
    output_pos = './ripe_data/'
    bin2mat(
        file_name=file_name, 
        output_pos=output_pos,
        num_ADCSamples = 196, 
        num_chirps = 255,
        num_RX=4
    )
    