### python3 .py 'gray file name' 'ss file name'

import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
import cv2
import copy
from pathlib import Path
from PIL import Image

args = sys.argv
gfname = f'input/{args[1]}points.txt'
ssfname = f'dataset/{args[2]}.png'

### depht読み込み
depth_points=[] #depth格納
dmap=np.zeros((480,640))
with open(gfname) as fp:
    for ln in fp:
        data = ln.strip().split(' ')
        depth_points.append(data);

### SS読み込み
img_ss = cv2.imread(ssfname, cv2.IMREAD_GRAYSCALE)
print('img_ss :', img_ss.shape)

### 出力画像作成
img_depth = np.zeros((480, 640), np.uint8)
img_cls = np.zeros((480, 640), np.uint8)
x=0
y=0
for ln in depth_points:
    x = int(ln[1])
    y = int(ln[0])
    dmap[x][y] = float(ln[2])
    img_depth[x][y] = 10    #depth着色
    img_cls[x][y] = 10

### clustering
h, w = img_ss.shape
car_idx = 20        #car color
for m,ln in enumerate(img_ss, 1):
    if m == 440:    #error
        break
    for n in range(w):
        if img_ss[m][n] == 8:   #抽出するクラス選択
            img_cls[m][n] = car_idx  #クラス着色
print('img_cls :', img_cls.shape)

### 物体検出
mom_ = []   #計算した重心を格納
img_dtc = copy.deepcopy(img_cls)
test = True
if test:
    _, threshold = cv2.threshold(img_dtc, car_idx-1, 30, cv2.THRESH_BINARY) #閾値
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #輪郭探索
    font = cv2.FONT_HERSHEY_DUPLEX  #フォント指定
    count = 0                       #物体数
    n = 0
    mom = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_dtc, [approx], 0, 10, 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        M = cv2.moments(cnt)        #モーメント

        xn = int(M['m10']/M['m00'])
        yn = int(M['m01']/M['m00'])
        mom = [xn, yn]
        mom_.append(mom)
        cv2.circle(img_dtc, (xn,yn), 2, (255,0,0), 1)
        n += 1

        ###重心周りのピクセルの平均距離算出
        mdp = 0         #出力するdepth
        count_dp = 0    #点の数
        sum_dp = 0
        for ln in range(5):
            for mn in range(5): #範囲指定
                dd = dmap[yn+ln][xn+mn]
                sum_dp += dd
                if dd > 0:
                    count_dp += 1
        if count_dp==0:
            # print('null dep')
            count_dp = 1
        mdp = sum_dp/count_dp
        print('mom mdp :',mom,mdp)
        ###

        if len(approx) > 10:
            count += 1
            cv2.putText(img_dtc, 'CAR', (x,y), font, 1, (25))

    print('Number of Car =', count)
    print('Moments :', mom_)

###出力画像の表示
fig = plt.figure(figsize=(8,16))
num_fig = 4
X=num_fig
Y=1

# for i in range(num_fig):

fig1 = 1
ax1 = fig.add_subplot(X, Y, fig1)
plt.imshow(img_ss)

fig2 = 2
ax2 = fig.add_subplot(X, Y, fig2)
plt.imshow(img_depth)

fig3 = 3
ax3 = fig.add_subplot(X, Y, fig3)
plt.imshow(img_cls)

fig4 = 4
ax4 = fig.add_subplot(X, Y, fig4)
plt.imshow(img_dtc)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

