import cv2
import numpy as np
import sys  # 导入sys模块


class DFS:
    length=0
    max_y=0
    min_y=0
    range_max=0
    vis=0
    length_min=0
    l=0
    h=0
    def __init__(self) -> None:
        sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

    def handler(self,imgaddr,outputaddr,filename):
        self.img = cv2.imread(imgaddr+filename+'.jpg',cv2.IMREAD_GRAYSCALE)
        self.range_max = self.img.shape[0]/4  #离中心最大距离
        self.length_min = 50
        self.l = self.img.shape[1]
        self.h = self.img.shape[0]
        self.max_y = 0
        self.min_y = 99999
        self.length = 0
        self.vis = np.zeros(self.img.shape)

        for x in range(self.l):
            for y in range(self.h):
                if(self.img[y][x]>=128 and not self.vis[y][x]):
                    self.length = 0
                    self.max_y = 0
                    self.min_y = 99999
                    self.dfs(x,y)
                
        cv2.imshow('',self.img)
        cv2.waitKey(0)
        cv2.imwrite(outputaddr+filename+'_2.jpg', self.img)

    def dfs(self,x,y):
        self.max_y = max(self.max_y,y)
        self.min_y = min(self.min_y,y)
        self.length = self.length + 1
        self.vis[y][x] = 1
        flag = 1
        flag2 = 0
        for xx,yy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            if(x+xx in range(self.l) and y+yy in range(self.h) and self.vis[y+yy][x+xx]==0 and self.img[y+yy][x+xx]>=128):
                if self.dfs(x+xx,y+yy):
                    flag2 = 1
                    self.img[y][x]=0
                flag=0
        if(flag2):
            return 1
        if(flag):
            # print(min_y,max_y)
            if self.length<self.length_min or self.max_y<self.range_max or self.min_y>self.range_max*3:
                self.img[y][x]=0
                return 1
        else:
            for xxx,yyy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                if(x+xxx in range(self.l) and y+yyy in range(self.h)):
                    self.img[y+yyy][x+xxx]=255
                    self.vis[y+yyy][x+xxx]=1
            return 0
                

if __name__=="__main__":
    dfs=DFS()
    dfs.handler('./output/figure/','./output/figure/','ft1')