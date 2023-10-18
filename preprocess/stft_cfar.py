import numpy as np
import cv2

def stft_cfar(org:np.ndarray):
    cfarvar=np.zeros_like(org)
    cfared=org.copy()
    dyn=dynaPfa(startPix=150,startVal=1e-30,endVal=1e-8)
    for i in range(org.shape[1]):
        tmp=cfared[:,i]
        cfarvar[:,i]=cfar_dynaPfa(org[:,i],dyn.genQ,15,3)
        tmp[tmp<cfarvar[:,i]]=0
        cfared[:,i]=tmp
    
    for x in np.nditer(cfared,op_flags=['readwrite']):
        if(x<140):
            x[...]=0
    return cfared

#cfar
def cfar_dynaPfa(signal:np.ndarray,Pfa,n_train:int = 10,n_guard:int = 3):
    N = len(signal)
    signal_cfar = np.zeros(N)
    # Pfa = 1e-5
    # n_train = 10  # 一侧的参考单元长度
    # n_guard = 3  # 一侧的保护单元的长度

    for i in range(N):
        if n_train + n_guard < i < N - n_train - n_guard:
            a = 2 * n_train * (Pfa(i) ** (-1 / (2 * n_train)) - 1)  # 门限因子
            nsum = np.sum(np.abs(signal[i - n_guard - n_train:i - n_guard]) ** 2) + np.sum(
                np.abs(signal[i + n_guard + 1:i + n_train + n_guard + 1]) ** 2)
            nsum = nsum / (2 * n_train)
            nsum = np.sqrt(a * nsum)
        elif i < n_train + n_guard + 1:  # 测量单元左侧参考单元内数据不足n_train个
            nsum = np.sum(np.abs(signal[i + n_guard + 1:i + n_train + n_guard + 1]) ** 2)
            n_tt = 0
            a = n_train * (Pfa(i) ** (-1 / n_train) - 1)  # 门限因子
            if i > n_guard + 1:  # 进入if表示测试单元左侧数据个数大于n_guard个，小于n_guard+n_train个
                nsum = nsum + np.sum(np.abs(signal[0:i - n_guard]))
                n_tt = i - n_guard - 1
                a = (n_train + n_tt) * (Pfa(i) ** (-1 / (n_train + n_tt)) - 1)
            nsum = nsum / (n_train + n_tt)
            nsum = np.sqrt(a * nsum)
        elif i > N - n_train - n_guard - 1:  # 测量单元右侧参考单元内数据不足n_train个
            nsum = np.sum(np.abs(signal[i - n_guard - n_train:i - n_guard]) ** 2)
            n_tt = 0
            a = n_train * (Pfa(i) ** (-1 / n_train) - 1) # 门限因子
            if i + n_guard < N+1:
                nsum = nsum + sum(abs(signal[i + n_guard : N]))
                n_tt = N - i - n_guard
                a = (n_train + n_tt) * (Pfa(i) ** (-1/(n_train + n_tt)) - 1)
            nsum = nsum/n_train + n_tt
            nsum = np.sqrt(a * nsum)
        signal_cfar[i] = nsum
    return signal_cfar

#动态虚警
class dynaPfa:
    mode='midPeak'
    scale=480
    startVal=1e-30
    startPix=5
    endVal=1e-20
    endPix=200
    arr=[]
    def __init__(self,mode:str='midPeak',scale:int=480,startVal:float=1e-30,startPix:int=5,endVal:float=1e-20,endPix:int=200) -> None:
        self.mode=mode
        self.startVal=startVal
        self.startPix=startPix
        self.endVal=endVal
        self.endPix=endPix
        self.scale=scale
        self.genPfaArr()

    def genPfaArr(self):
        for i in range(self.scale):
            self.arr.append(self.PfaGen(i))
    def genQ(self,n:int):
        return self.arr[n]
    def PfaGen(self,n:int):
        if(self.mode=='midPeak'):
            if(n<=self.startPix or n>=self.scale-self.startPix):
                return self.startVal
            elif(n>self.startPix and n<self.endPix):
                return (self.endVal-self.startVal)/(self.endPix-self.startPix)*(n-self.startPix)
            elif(n>self.scale/2-self.endPix+self.scale/2 and n<self.scale-self.startPix):
                return -(self.endVal-self.startVal)/(self.endPix-self.startPix)*(n-self.scale/2+self.endPix-self.scale/2)+self.endVal
            elif(n>=self.endPix and n<=self.scale/2+self.scale/2-self.endPix):
                return self.endVal