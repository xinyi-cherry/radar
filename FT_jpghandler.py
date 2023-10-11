import numpy as np
import matplotlib.pyplot as plt
import cv2 

ISDEBUG = True

def mainEdgeGetter(inImg:np.ndarray)->np.ndarray:
    #dbg
    dbg=debugPlot(ISDEBUG)

    tar=inImg.copy()
    dbg.recFigs('Original',tar)
    ker1=np.array([[0,1,0],[1,-1.5,1],[0,1,0]]) #低通卷积核
    gus1=cv2.filter2D(inImg,-1,ker1) #低通获取低频噪声
    dbg.recFigs('LPF Filtered',gus1)
    tar=tar-gus1 #去除高频噪声
    dbg.recFigs('Ori minus Low freq',tar)
    tar[np.where(tar<140)]=0 #去除值较低的背景噪声
    dbg.recFigs('Del lower values',tar)
    ker2=np.array([[0,1,0],[1,-4,1],[0,1,0]]) #拉普拉斯算子用于检测边缘
    tar=cv2.blur(tar,(1,1)) #平滑一格以创建可供识别的边缘
    dbg.recFigs('Blur its Edge',tar)
    tar=cv2.filter2D(tar,-1,ker2) #边缘检测以获取边缘
    dbg.recFigs('Get its Edge',tar)
    _,tar=cv2.threshold(tar,10,255,cv2.THRESH_BINARY) #二值化以检测边缘
    dbg.recFigs('Tresh_binary its Edge',tar)
    # dbg.plot('subplt')
    # dbg.plot('saveFig','./output/jpghandlerOutput/')
    dbg.plot('showFigSingle')
    return tar

class debugPlot:
    isdbg=False
    figsSave=dict()
    figConf=dict()
    def __init__(self,isdbg:bool) -> None:
        self.isdbg=isdbg

    def recFigs(self,name:str,figArr:np.ndarray,cmap=-1):
        if(self.isdbg==False): return
        self.figsSave[name]=figArr.copy()
        self.figConf[name]=cmap
    
    def plot(self,mode:str,saveDict:str=''):
        """
         saveDict 仅适用于mode=="saveFig" 结尾要有'/'
        """
        if(self.isdbg==False): return
        lth=len(self.figsSave)
        if(mode=='subplt'):
            wd=int(np.floor(np.sqrt(lth)))
            ht=int(np.ceil(lth/wd))

            plt.figure()
            for i,subfig in enumerate(self.figsSave.keys()):
                plt.subplot(ht,wd,i+1)
                if(self.figConf[subfig]!=-1):
                    plt.imshow(self.figsSave[subfig],cmap=self.figConf[subfig])
                else: plt.imshow(self.figsSave[subfig])
                plt.title(subfig)
            plt.show()
        elif(mode=='saveFig'):
            # 定义颜色映射
            for i,subfig in enumerate(self.figsSave.keys()):
                cv2.imwrite(saveDict+subfig+'_c.jpg',self.figsSave[subfig])
            #TODO 多图直出
        elif(mode=='showFigSingle'):
            for i,subfig in enumerate(self.figsSave.keys()):
                plt.figure(i)
                if(self.figConf[subfig]!=-1):
                    plt.imshow(self.figsSave[subfig],cmap=self.figConf[subfig])
                else: plt.imshow(self.figsSave[subfig])
                plt.title(subfig)
                plt.show()
            



if __name__ == '__main__':
    img=cv2.imread('./output/figs/data1_1_Raw_0_FT_512_1.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    rt=mainEdgeGetter(img)