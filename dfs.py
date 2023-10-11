import cv2
import numpy as np
import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)

range_max = img.shape[0]/4  #离中心最大距离
length_min = 50

print(img)

l = img.shape[1]
h = img.shape[0]

max_y = 0
min_y = 99999
length = 0
vis = np.zeros(img.shape)

def dfs(x,y):
    global length
    global max_y
    global min_y
    max_y = max(max_y,y)
    min_y = min(min_y,y)
    length = length + 1
    vis[y][x] = 1
    flag = 1
    flag2 = 0
    for xx,yy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
        if(x+xx in range(l) and y+yy in range(h) and vis[y+yy][x+xx]==0 and img[y+yy][x+xx]>=128):
            if dfs(x+xx,y+yy):
                flag2 = 1
                img[y][x]=0
            flag=0
    if(flag2):
        return 1
    if(flag):
        print(min_y,max_y)
        if length<length_min or max_y<range_max or min_y>range_max*3:
            img[y][x]=0
            return 1
    else:
        for xxx,yyy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            if(x+xxx in range(l) and y+yyy in range(h)):
                img[y+yyy][x+xxx]=255
                vis[y+yyy][x+xxx]=1
        return 0
                
                

for x in range(l):
    for y in range(h):
        if(img[y][x]>=128 and not vis[y][x]):
            length = 0
            max_y = 0
            min_y = 99999
            dfs(x,y)
        
cv2.imshow('',img)


cv2.waitKey(0)
cv2.imwrite('modified.jpg', img)
print(img.shape[0])