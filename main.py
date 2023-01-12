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
depth_points=[]                         #depth格納
dmap=np.zeros((480,640))                #480*640
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
    img_depth[x][y] = 10        #depth着色
    img_cls[x][y] = 10          #

### clustering
h, w = img_ss.shape
car_color = 20        #car color
for m,ln in enumerate(img_ss, 1):
    if m == 440:    #error
        break
    for n in range(w):
        if img_ss[m][n] == 8:           #抽出するクラス選択
            img_cls[m][n] = car_color     #クラス着色
print('img_cls :', img_cls.shape)

### 物体検出
carDepthPoints = [] #segした車のdepth保存
moms = []           #計算したすべての重心を格納
img_dtc = copy.deepcopy(img_cls)
test = True
if test:
    _, threshold = cv2.threshold(img_dtc, car_color-1, car_color+1, cv2.THRESH_BINARY)              #閾値
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #輪郭探索

    font = cv2.FONT_HERSHEY_DUPLEX  #フォント指定
    count = 0                       #物体数
    count_mo = 0                    #重心数
    n = 0
    mom = []                        #計算した重心を格納
    mndp = 0                        #seg内の全depthの平均

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_dtc, [approx], 0, 155, 1)                           #輪郭描画
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        M = cv2.moments(cnt)        #モーメント
        count_mo += 1

        print('\nstart moment =', count_mo)

        xn = int(M['m10']/M['m00'])
        yn = int(M['m01']/M['m00'])
        mom = [count_mo, xn, yn]
        moms.append(mom)
        cv2.circle(img_dtc, (xn,yn), 2, (255,0,0), 1)
        n += 1

        ### bb作成
        bbx, bby, bbw, bbh = cv2.boundingRect(cnt)
        cv2.rectangle(img_dtc, (bbx,bby), (bbx+bbw-1,bby+bbh-1), car_color-3, 1)

        ### bb内のcarのdepthの平均算出
        mndp=0
        count_dp=0    #点の数
        sum_dp=0
        max_range=40
        for xm in range(bbx, bbx+bbw-1):
            for ym in range(bby, bby+bbh-1):
                try:
                    if img_dtc[ym][xm]==car_color and dmap[xm][ym]<=max_range:  # carかつmax_range以内
                        dd = dmap[xm][ym]
                        sum_dp += dd
                        if dd > 0:
                            count_dp += 1
                except IndexError:
                    pass

            # try:
            #     print(bbx+bbw/2,ym,img_dtc[bbx+bbw/2][xm],car_color)
            # except IndexError:
            #     pass

        if count_dp==0:
            count_dp = 1

        print(f'depth data : {sum_dp:8.2f} {count_dp:3}')
        mndp = sum_dp/count_dp


        ### 重心周りのピクセルの平均depth算出
        mdp = 0         #出力するdepth
        count_dp = 0    #点の数
        sum_dp = 0
        for ln in range(5):
            for mn in range(5):         #範囲指定
                dd = dmap[yn+ln][xn+mn] #depth取得
                sum_dp += dd
                if dd > 0:              #点が存在したらcount++
                    count_dp += 1
        if count_dp==0:                 #重心周りに点がない場合
            # print('null dep')
            count_dp = 1
        mdp = sum_dp/count_dp
        ###

        print('mom :', mom)
        print(f'momdp meandp : {mdp:3.1f} {mndp:3.1f}')

        if len(approx) > 10:
            count += 1
            cv2.putText(img_dtc, 'CAR', (x,y), font, 0.5, (255))

    print('Number of Car =', count)
    # print('Moments :', moms)

###出力画像の表示
size=10
fig = plt.figure(figsize=(size*1.6,size))
num_fig = 4
X=2
Y=num_fig/X

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

