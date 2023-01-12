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

### depht読み込み
def input_depth_points(gfname):
    depth_points=[]                         #depth格納

    with open(gfname) as fp:
        for ln in fp:
            data = ln.strip().split(' ')
            depth_points.append(data);

    return depth_points
### -> depth点群


### SS読み込み
def input_SS_image(ssfname):
    img_ss = cv2.imread(ssfname, cv2.IMREAD_GRAYSCALE)
    print('img_ss :', img_ss.shape)

    return img_ss
### SS画像

### 出力画像作成
def make_output_image(depth_points):
    dmap      = np.zeros((480,640))
    img_depth = np.zeros((480, 640), np.uint8)
    xd=0
    yd=0
    cDepth=0

    for ln in depth_points:
        xd = int(ln[1])
        yd = int(ln[0])
        dmap[xd][yd] = float(ln[2])
        cDepth = 10 #+ int(float(ln[2]))
        img_depth[xd][yd] = cDepth        #depth着色

    return img_depth, dmap
### -> depth image, depth map

### clustering
def clustering(img_ss, img_depth):
    h, w = img_ss.shape
    img_cls = img_depth

    for m,ln in enumerate(img_ss, 1):
        if m == 440:        #error
            break
        for n in range(w):
            if img_ss[m][n] == 8:               #抽出するクラス選択
                img_cls[m][n] = car_color_       #クラス着色

    print('img_cls :', img_cls.shape)

    return img_cls
### -> clustering image

### bb内のcarのdepthの平均算出
def mean_depth_bbox(img_dtc, dmap, cnt):
    ### bb作成
    bbx, bby, bbw, bbh = cv2.boundingRect(cnt)
    cv2.rectangle(img_dtc, (bbx,bby), (bbx+bbw-1,bby+bbh-1), car_color_-3, 1)

    ### bb内のcarのdepthの平均算出
    mndp        = 0          #平均depth
    count_dp_b  = 0    #点の数
    sum_dp_b    = 0
    max_range   = 40    #考慮する最大距離

    for xm in range(bbx, bbx+bbw-1):
        for ym in range(bby, bby+bbh-1):
            try:    #配列外ならスキップ
                if img_dtc[ym][xm]==car_color_ and dmap[xm][ym]<=max_range:  # carかつmax_range以内
                    dd = dmap[xm][ym]
                    sum_dp_b += dd
                    if dd > 0:
                        count_dp_b += 1
            except IndexError:
                pass

    if count_dp_b==0:
        count_dp_b = 1

    mndp = sum_dp_b/count_dp_b

    print(f' depth data : {sum_dp_b:8.2f} {count_dp_b:3}')
    ###

    return mndp
### -> 平均depth

### 重心周りのピクセルの平均depth計算
def mean_depth_mom(mom, dmap):
    mdp = 0         #出力するdepth
    count_dp = 0    #点の数
    sum_dp = 0
    xn = mom[0]
    yn = mom[1]

    for ln in range(5):
        for mn in range(5):         #範囲指定
            dd = dmap[yn+ln][xn+mn] #depth取得
            sum_dp += dd
            if dd > 0:              #点が存在したらcount++
                count_dp += 1
    if count_dp==0:                 #重心周りに点がない場合
        count_dp = 1
    mdp = sum_dp/count_dp

    return mdp
###

###
def isCar(img_dtc, cnt, mom, count):
    font = cv2.FONT_HERSHEY_DUPLEX  #フォント指定

    if len(cnt) > 30:
        count += 1
        print(f' car number {count}')
        cv2.putText(img_dtc, f'CAR {count}', (mom[0],mom[1]), font, 0.5, (255))

    return count
###

### 物体検出
def detection(img_cls, dmap):
    carDepthPoints = []                 #segした車のdepth保存
    img_dtc = copy.deepcopy(img_cls)    #出力する画像
    font = cv2.FONT_HERSHEY_DUPLEX      #フォント指定

    count    = 0                        #物体数
    count_mo = 0                        #重心数
    n        = 0
    mndp     = 0                        #seg内の全depthの平均

    mom      = []                       #計算した重心を格納
    moms     = []                       #計算したすべての重心を格納

    ### 輪郭探索
    _, threshold = cv2.threshold(img_dtc, car_color_-1, car_color_+1, cv2.THRESH_BINARY)        #閾値
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #輪郭探索

    ### それぞれの輪郭に対して処理
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_dtc, [approx], 0, 155, 1)                           #輪郭描画
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        M = cv2.moments(cnt)        #モーメント
        count_mo += 1

        print('\nstart moment =', count_mo)

        try:
            xn = int(M['m10']/M['m00'])
            yn = int(M['m01']/M['m00'])
            mom = [xn, yn]
            moms.append(mom)
            cv2.circle(img_dtc, (xn,yn), 2, (255,0,0), 1)
            n += 1

            ### bb内のcarのdepthの平均算出
            mndp = mean_depth_bbox(img_dtc, dmap, cnt)

            ### 重心周りのピクセルの平均depth算出
            mdp = mean_depth_mom(mom, dmap)

            ### 輪郭のピクセル数がn以上でクラス判定
            count = isCar(img_dtc, cnt, mom, count)

            print(' contours :', len(cnt))                       #輪郭のピクセル数
            print(' mom :', mom)
            print(f' momdp meandp : {mdp:3.1f} {mndp:3.1f}')

        except ZeroDivisionError:
            pass

    print('\nNumber of Car =', count)
    print('Moments :', moms)

    return moms, img_dtc
### -> 重心とdepthデータ，処理後の画像

### 出力画像の表示
def show_result(img_ss, img_depth, img_cls, img_dtc):
    size=15
    fig = plt.figure(figsize=(size*1.6,size))
    num_fig = 4
    X=2
    Y=num_fig/X

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
###


### para
car_color_ = 20

def main():
    ### file name
    args = sys.argv
    gfname = f'input/{args[1]}points.txt'
    ssfname = f'dataset/{args[2]}.png'
    ### para

    ### 処理
    depth_points         = input_depth_points(gfname)
    img_ss               = input_SS_image(ssfname)
    img_depth, depth_map = make_output_image(depth_points)
    img_cls              = clustering(img_ss, img_depth)
    moms_, img_dtc       = detection(img_cls, depth_map)

    show_result(img_ss, img_depth, img_cls, img_dtc)

if __name__ == "__main__":
    main()

