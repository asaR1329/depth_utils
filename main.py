### python3 .py 'gray file name' 'ss file name' '時間幅'
### 230118

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import re
import cv2
import copy
import time
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
    # print('img_ss :', img_ss.shape)

    return img_ss
### SS画像

### 出力画像作成
def make_output_image(depth_points):
    dmap      = np.zeros((480,640))
    img_depth = np.zeros((480, 640), np.uint8)
    xd = 0
    yd = 0
    cDepth =0

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
    img_cls = copy.deepcopy(img_depth)

    for m,ln in enumerate(img_ss, 1):
        if m == 440:        #error
            break
        for n in range(w):
            if img_ss[m][n] == class_idx_:               #抽出するクラス選択
                img_cls[m][n] = car_color_       #クラス着色

    # print('img_cls :', img_cls.shape)

    return img_cls
### -> clustering image

### bb内のcarのdepthの平均算出
def mean_depth_bbox(img_dtc, dmap, cnt):
    ### bb作成
    bbx, bby, bbw, bbh = cv2.boundingRect(cnt)
    cv2.rectangle(img_dtc, (bbx,bby), (bbx+bbw-1,bby+bbh-1), car_color_-3, 1)

    ### bb内のcarのdepthの平均算出
    mndp        = 0 #平均depth
    count_dp_b  = 0 #点の数
    sum_dp_b    = 0 #depthの合計
    global watcher
    for xm in range(bbx, bbx+bbw-1):
        for ym in range(bby, bby+bbh-1):
            try:    #配列外ならスキップ
                dd = dmap[xm][ym]
                if img_dtc[ym][xm]==car_color_ and dd<=max_range_: # 選択したクラスかつ max_range_以内の点を考慮
                    sum_dp_b += dd
                    if dd > 0:  #dephtが存在したら数える
                        count_dp_b += 1
                if img_dtc[ym][xm]==car_color_ and dmap[xm][ym]>0: watcher.append([xm, dmap[xm][ym]])
            except IndexError:
                pass

    if count_dp_b==0:   # 0わり
        count_dp_b = 1

    mndp = sum_dp_b/count_dp_b

    # if len(watcher)>=30: show_scatter(watcher)
    watcher=[]
    print(f' depth data : sum={sum_dp_b:8.2f} count={count_dp_b:3}')
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

    length = 5
    for ln in range(length):
        for mn in range(length):         #範囲指定
            dd = dmap[yn+ln][xn+mn] #depth取得
            sum_dp += dd
            if dd > 0:              #点が存在したらcount++
                count_dp += 1
    if count_dp==0:                 #重心周りに点がない場合
        count_dp = 1
    mdp = sum_dp/count_dp

    return mdp
###

### 輪郭の数が一定以上で物体判定
def isCar(img_dtc, cnt, mom, min_cluster_size_, count):
    font = cv2.FONT_HERSHEY_DUPLEX  #フォント指定

    if len(cnt) > min_cluster_size_:
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

        print('\nmoment :', count_mo)

        if len(cnt)>=min_cluster_size_:
            try:
                xn = int(M['m10']/M['m00'])
                yn = int(M['m01']/M['m00'])
                mom = [xn, yn]
                cv2.circle(img_dtc, (xn,yn), 2, (255,0,0), 1)
                n += 1

                ### bb内のcarのdepthの平均算出
                mndp = mean_depth_bbox(img_dtc, dmap, cnt)

                ### 重心周りのピクセルの平均depth算出
                mdp = mean_depth_mom(mom, dmap) #使用しない

                ### 輪郭のピクセル数がn以上でクラス判定
                count = isCar(img_dtc, cnt, mom, min_cluster_size_, count)

                print(f' contours length:', len(cnt))                       #輪郭のピクセル数
                print(f' mom :', mom)
                print(f' momdp meandp : {mdp:3.1f} {mndp:3.1f}')

                mom.append(mndp)
                moms.append(mom)

            except ZeroDivisionError:
                pass
        else:
            print(f'not cluster')

    print('\nNumber of Car =', count)
    print('Moments :', moms)

    return moms, img_dtc
### -> 重心とdepthデータ，処理後の画像

### 出力画像の表示
def show_images(img_ss, img_depth, img_cls, img_dtc):
    windowSize=8

    ### 画像の表示
    fig = plt.figure(figsize=(windowSize*1.6,windowSize))
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
    ###

    plt.show(block=False)
    key = cv2.waitKey(0)
    if key==ord('d'): plt.close(fig)
###

### 重心の軌跡表示
def show_tracer(tracer_wID):
    windowSize = 8
    figMom = plt.figure(figsize=(windowSize,windowSize))
    axTr = figMom.add_subplot(1,1,1)
    axTr.set_title('Tracer')
    axTr.set_xlim(0,640)
    # axTr.set_ylim(0,480) #y
    axTr.set_ylim(-1,100) #depth
    plt.xlabel('x axis')
    plt.ylabel('depth')

    dx, dy = [], []
    op_x, op_y = [], []

    try:
        for ID in range(0,10): # 全IDに対して
            dx, dy = [], []
            for i in range(len(tracer_wID)):
                for j in range(len(tracer_wID[i])):
                    if ID == tracer_wID[i][j][0]:
                        dx.append(tracer_wID[i][j][1])
                        # dy.append(tracer_wID[i][j][2])    # yaxis
                        dy.append(tracer_wID[i][j][3])      # depth

            op_x.append(dx)
            op_y.append(dy)

            color = cm.tab20(ID)
            plt.scatter(op_x[ID], op_y[ID], label=f'id={ID}', color=color, s=5)
            plt.legend()
            plt.plot(op_x[ID],op_y[ID],color=color)

    except IndexError:
        pass

        ### 重心位置の表示
    plt.show()
    ###
    return 0
###

###
def show_scatter(data):
    windowSize = 8
    figw = plt.figure(figsize=(windowSize,windowSize))
    axW = figw.add_subplot(1,1,1)
    axW.set_title('watch')
    axW.set_xlim(0,640)
    axW.set_ylim(-1,200) #depth
    plt.xlabel('x axis')
    plt.ylabel('depth')

    dx, dy = [], []
    op_x, op_y = [], []

    for nn in data:
        dx.append(nn[0])
        dy.append(nn[1])

    op_x.append(dx)
    op_y.append(dy)

    plt.scatter(op_x, op_y, s=5)
    plt.show()
###

### 1frameの処理まとめる
def estimateMoms(gfname, ssfname):
    depth_points         = input_depth_points(gfname)
    img_ss               = input_SS_image(ssfname)
    img_depth, depth_map = make_output_image(depth_points)
    img_cls              = clustering(img_ss, img_depth)
    moms, img_dtc        = detection(img_cls, depth_map)

    return moms, img_ss, img_depth, img_cls, img_dtc
### 1フレームの重心，出力画像

### 重心距離から同一か判定
def decide_mom_id(tracer):
    momID = 0
    maxID = 0           #最大のID
    data = []           # 1frameの重心とID格納
    tracer_wID = []     #IDつきの重心

    a = np.array([0,0])
    b = np.array([0,0])
    distance = 0

    np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
    print(f'\n---all car moment points---')
    print(*tracer, sep='\n')

    for i in range(len(tracer)):             # 全フレームに対して
        momID = 0
        data = []

        if i==0:    # 最初のフレームに対して
            for j in range(len(tracer[i])):
                momID+=1
                a = np.array(tracer[i][j])
                data.append(np.append(momID,a))
        else:
            for j in range(len(tracer[i])):     # i   の全重心
                a = np.array(tracer[i][j])          # 現在の重心

                for k in range(len(tracer[i-1])):   # i-1 の全重心
                    try:
                        b = np.array([tracer_wID[i-1][k][1], tracer_wID[i-1][k][2], tracer_wID[i-1][k][3]])    # 1frame前の重心
                        distance = np.linalg.norm(b-a)
                        if distance<=tolerance:
                            momID = tracer_wID[i-1][k][0]
                            data.append(np.append(momID,a))
                            if maxID <= momID:
                                maxID = momID
                            break
                    except IndexError:
                        pass

        tracer_wID.append(data)     # 1frameの重心データ格納

    return tracer_wID
###

### 一定時間の重心の座標算出
def make_tracer(fname, tm_):
    tracer = []
    ### tm_ frame分実行
    try:

        for ll in range(tm_):
            print(f'\n---estimate frame={ll}---')
            ###ファイル名調整 fname:11500
            f1name = '000000'
            f2name = '000000'

            f1  = int(fname) + ll*50   #11500+50n
            f1name = f1name[:6-len(str(f1))] + str(f1) #011500
            f2  = int(int(f1)*2/10)  #2300+
            f2name = f2name[:6-len(str(f2))] + str(f2) #002300

            gfname  = f'input/{f1name}points.txt'
            ssfname = f'dataset/{f2name}.png'
            print(f' {gfname}, {ssfname}')
            ###

            ### 重心と画像出力
            moms, img_ss, img_depth, img_cls, img_dtc = estimateMoms(gfname, ssfname)
            tracer.append(moms)
            if ll%number_image_==0: show_images(img_ss, img_depth, img_cls, img_dtc) # n frameごとに
            ###

    except IndexError:
        print('\n===tracer (index error)===')
        print(*tracer, sep='\n')

    ### id配布
    tracer_wID = decide_mom_id(tracer)
    print('\n=== tracer with ID ===')
    print('frame    | ID | x | y | depth |')
    for i in range(len(tracer_wID)):
        for j in range(len(tracer_wID[i])):
            print(f'frame ={i:3d} ID ={tracer_wID[i][j][0]:2.0f} x = {tracer_wID[i][j][1]:3.0f} y = {tracer_wID[i][j][2]:3.0f} depth = {tracer_wID[i][j][3]:.4g}')
    ###

    ### watch
    # show_scatter(watcher)
    ###


    ### 軌跡表示
    show_tracer(tracer_wID)

    return 0
###

### parameter
class_idx_          =  8    # 抽出するクラス 3:human 8:car
car_color_          = 20    # クラスタリングするときの色
min_cluster_size_   = 20    # クラスタリングする点の最少数
tolerance           = 50    # 同一物体と許容する距離
max_range_          = 70    # 考慮する最大距離
tm_                 = 10    # 実行フレーム数
number_image_       =  4    # nフレームごとに画像出力
watcher             = []    # テスト用
###
def main():
    ### file name
    args = sys.argv
    fname = '000000'
    fname = args[1]
    timeWidth = args[2]
    ### param
    tm_ = int(timeWidth)
    np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

    ### 処理
    make_tracer(fname, tm_)
###

if __name__ == "__main__":
    main()

