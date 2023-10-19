from bin2mat_new import bin2matNew
import scipy.io as io
from RD_cross_pad import cal_RD
from FT_PT import phase
from  multiprocessing import Process,Pool
import os
import time
import stft_cfar


class RadarWorkFlow:
    # global vars
    options={
        'num_ADCSamples': 128,
        'num_chirps': 255,
        'num_frames': 96,
        'AT_num_RX':8
    }
    binOriginRootPosition="./raw_data/"
    matSaveRootPosition='./ripe_data/'
    figOutputPosition='../output/figs'
    threads=2
    def __init__(self,
                 options={
                    'num_ADCSamples': 128,
                    'num_chirps': 255,
                    'num_frames': 96,
                    'AT_num_RX':8
                    },
                 binOriginRootPosition:str="./raw_data/",
                 matSaveRootPosition:str='./ripe_data/',
                 figOutputPosition:str='../output/figs/'
                ) -> None:
        self.options=options
        self.binOriginRootPosition=binOriginRootPosition
        self.matSaveRootPosition=matSaveRootPosition
        self.figOutputPosition=figOutputPosition
    #单一文件的处理
    def singleDataHandler(self,isFromNewBINfile:bool=True,
                    binfileName:str='-1',
                    matFileName:str='-1',
                    isATorRD:str='AT'):
        '''
            isFromNewBINfile:是否从原始bin中开始处理
            注意：文件名全不要带拓展名
        '''
        # if(isFromNewBINfile):
        #     assert(binfileName=='-1','未输入bin文件名')
        # else:
        #     assert(matFileName=='-1','未输入mat文件名')
        currentbfName=''
        if(isFromNewBINfile):
            bin2matNew(binfileName,self.binOriginRootPosition,self.matSaveRootPosition,num_ADC=self.options['num_ADCSamples'],num_chirps_pframe=self.options['num_chirps'],num_RX=self.options['AT_num_RX'])
            currentbfName=binfileName
        else:
            currentbfName=matFileName
        
        # '_test.mat'结尾
        adcData = io.loadmat(self.matSaveRootPosition+currentbfName+'_test.mat')
        adcData = adcData['adcData']

        #cal_RD(file_num=1, file_name=currentbfName, adcData=adcData,
        #        num_ADCSamples=self.options['num_ADCSamples'],num_chirps=self.options['num_chirps'],num_frames=self.options['num_frames'])
        phase(type=1,file_name=currentbfName,file_num=1,adcData=adcData,file_position=self.figOutputPosition,
                num_ADCSamples=self.options['num_ADCSamples'],num_chirps=self.options['num_chirps'],num_frame=self.options['num_frames'])
        stft_cfar.stft_cfar_warp(currentbfName,self.figOutputPosition)
    #并行处理
    def parallelProcess(self,threads:int=2):
        """
            从原始数据开始处理
        """
        self.threads=threads
        print('\033[31m MainThread@Message::\033[0m Create multiprocessing pool with %s process.' % str(self.threads))
        pool=Pool(self.threads)
        file_names = [f[0:-4] for f in os.listdir(self.binOriginRootPosition) if f.endswith('.bin')]
        for i in range(len(file_names)):
            pool.apply_async(func=self.singleDataHandler,args=(True,file_names[i],'-1','RD'))
        pool.close()
        pool.join()
        print('\033[31m MainThread@Message::\033[0m Finish processing.')


opts={
    'num_ADCSamples': 128,
    'num_chirps': 255,
    'num_frames': 96,
    'AT_num_RX':8
}

if __name__=='__main__':
    start_time = time.time()
    wf=RadarWorkFlow(opts)
    wf.parallelProcess(2)
    end_time = time.time()
    print("\033[31m MainThread@Message::\033[0m time used:", end_time - start_time)
# wf.singleDataHandler(isFromNewBINfile=True,binfileName='slope180_chirp255_ababa2_Raw_0',isATorRD='RD')


