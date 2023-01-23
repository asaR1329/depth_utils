### python3 .py 'gray file name' 'ss file name' '時間幅'
### 230122

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import csv
import cv2
import glob
import math
import matplotlib
import os
import re
import rospy
import sys
import time
import traceback
import yaml
from pathlib import Path
from PIL import Image
from nav_msgs.msg import Path
from filterpy.kalman import predict, update
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

matplotlib.use('TkAgg')

class depthPoint:
    x = 0
    y = 0
    d = 0

    def __init__(self, x, y, depth):
        self.x = x
        self.y = y
        self.d = depth

### depht読み込み
def input_depth_points(gfname):
    depth_points=[]                         #depth格納
    count = 0

    with open(gfname) as fp:
        for ln in fp:
            data = ln.strip().split(' ')
            depth_points.append(data);
            count += 1

    print(f' number of points = {count}')

    return depth_points
### -> depth点群


### SS読み込み
def input_SS_image(ssfname):
    img_ss = cv2.imread(ssfname, cv2.IMREAD_GRAYSCALE)

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
            if img_ss[m][n] == class_idx_:       #抽出するクラス選択
                img_cls[m][n] = obj_color_       #クラス着色

    return img_cls
### -> clustering image

### bb内のobjのdepthの平均算出
def mean_depth_bbox(img_dtc, dmap, cnt):
    ### bb作成
    bbx, bby, bbw, bbh = cv2.boundingRect(cnt)
    cv2.rectangle(img_dtc, (bbx,bby), (bbx+bbw-1,bby+bbh-1), obj_color_-3, 1)

    ### bb内のtarget classのdepth平均算出
    mndp        = 0 #平均depth
    count_dp_b  = 0 #点の数
    sum_dp_b    = 0 #depthの合計
    depth_clusters = [] # クラスター群
    dp = depthPoint(0, 0, 0)  #depth_cluster内の点
    tr_count = 0
    isClustering = False
    global watcher

    ### bb内の全pixelに対して
    # print(f' sampling bb')
    for xm in range(bbx, bbx+bbw-1):
        for ym in range(bby, bby+bbh-1):

            try:    #配列外ならスキップ
                dd = dmap[xm][ym] # ある点のdepthを抽出

                ###
                if dd!=0 and dd!=200: #depthが取れた点
                    # watcher.append([xm, dd])
                    if img_dtc[ym][xm]==obj_color_ and dd<=max_range_: # 選択したクラスかつ max_range_以内の点を考慮
                        ### depthクラスター作成
                        dp = depthPoint(xm,ym,dd)
                        isClustering = False
                        if len(depth_clusters)==0:
                            depth_clusters.append([[dp.x, dp.y, dp.d]])
                            print(f' make cluster d={dp.d:.2f}')
                        else:
                            for i in range(len(depth_clusters)): # [[[xyz],[xyz],...],[...]] クラスター群探索
                                for j in range(len(depth_clusters[i])): # [[xyz],[xyz],...]
                                    if abs(depth_clusters[i][j][2]-dp.d)<=tolerance_dc_ and not(isClustering): # クラスタ間の距離 tolerance_dc_
                                        depth_clusters[i].append([dp.x, dp.y, dp.d])
                                        isClustering = True
                            if isClustering==False: # 既存のクラスターに入らなかったら新しく作る
                                depth_clusters.append([[dp.x, dp.y, dp.d]])
                                print(f' make cluster d={dp.d:.2f}')
                        ###
                        # 平均depth算出
                        sum_dp_b += dd
                        count_dp_b += 1 #dephtが存在したら数える
                    ###
                # else:
                #     tr_count += 1
                ###

            except IndexError:
                pass
    ### bb内の探索終了

    # show_scatter(watcher)
    watcher=[]

    if count_dp_b==0:   # not /0
        count_dp_b = 1

    mndp = sum_dp_b/count_dp_b # depth ave 計算

    ### 最大数のクラスターを探してその平均をとる
    if count_dp_b!=1: # depthが取れているとき
        max_idx = 0         # 最大数のインデックス
        sub_max_idx = 0     # 2番め
        max_count = 0       # 最大数
        sub_max_count = 0   # 2
        sum_depth = 0
        ave_depth = 0
        # depth clusterのprint
        for i in range(len(depth_clusters)):
            for j in range(len(depth_clusters[i])):
                ave_depth += depth_clusters[i][j][2]    # クラスターの平均depthを出す

            if len(depth_clusters[i]) > max_count:
                    sub_max_idx = max_idx               # most => sub
                    max_idx = i                         # 最大数のインデックスを保存
                    max_count = len(depth_clusters[i])  # 数を保存

            ave_depth /= len(depth_clusters[i])
            print(f'  cluster{i} = count:{len(depth_clusters[i]):3d} depth:{ave_depth:.2f}')
            ave_depth = 0

            # 平均depthの差が小さい && 少ない方のdepthが小さいとき．．．

        for i in range(len(depth_clusters[max_idx])):
            sum_depth += depth_clusters[max_idx][i][2]
        mndp = sum_depth/max_count  # ave depth 計算
    else:
        print(' no depth"')
    ###

    print(f' target class depth data : sum={sum_dp_b:8.2f} count={count_dp_b:3}')
    # print(f' cannot est points : {tr_count:4d}')
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
        for mn in range(length):    #範囲指定
            dd = dmap[yn+ln][xn+mn] #depth取得
            sum_dp += dd
            if dd > 0:              #点が存在したらcount++
                count_dp += 1
    if count_dp==0: count_dp = 1    #重心周りに点がない場合
    mdp = sum_dp/count_dp

    return mdp
###

### 輪郭の数が一定以上で物体判定
def isCar(img_dtc, cnt, mom, min_cluster_size_, count):
    font = cv2.FONT_HERSHEY_DUPLEX  #フォント指定

    if len(cnt) > min_cluster_size_:
        count += 1
        print(f' obj number {count}')
        cv2.putText(img_dtc, f'Obj {count}', (mom[0],mom[1]), font, 0.5, (255))

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
    _, threshold = cv2.threshold(img_dtc, obj_color_-1, obj_color_+1, cv2.THRESH_BINARY)        #閾値
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
                print(f' momdp meandp : {mdp:5.21f} {mndp:5.21f}')

                if mndp!=0: # 平均depthが０でないとき
                # if True: # 平均depthが０でないとき
                    mom.append(mndp)
                    moms.append(mom)

            except ZeroDivisionError:
                pass
        else:
            print(f'not cluster')

    print('\nNumber of Object =', count)
    print('Moments :', moms)

    return moms, img_dtc
### -> 重心とdepthデータ，処理後の画像

### カメラ座標系からLiDAR座標系への変換
def img2world(moms):
    ### image2cam
    calib2cam = np.zeros((3,3))
    with open(calib_path) as fp:
        calib_data = yaml.safe_load(fp)
        camera_info = calib_data['intrinsics']['camRect0']['camera_matrix']
        calib2cam = np.array(
                    [[camera_info[0], 0, camera_info[2]],
                     [ 0, camera_info[1], camera_info[3]],
                     [ 0, 0, 1]]
                    )

    ### cam2world
    calib2LiDAR = np.zeros((4,4))
    with open(calib_path_lidar) as fp:
        calib_data = yaml.safe_load(fp)
        c2l = calib_data['T_lidar_camRect1']

    ###convert
    data = copy.deepcopy(moms) # copy
    for i in range(len(data)):
        print(f' pre:x = {moms[i][0]:3.0f} y = {moms[i][1]:3.0f} d = {moms[i][2]:5.2f}')
        data[i] = np.array(data[i])
        data[i][2] = 1          # [xi, yi, 1]
        data[i] = np.dot(np.linalg.inv(calib2cam), data[i])
        #
        moms[i][0] = data[i][0]
        moms[i][1] = data[i][1] # [X, Y, 1]
        # *= depth
        moms[i][0] *= moms[i][2]
        moms[i][1] *= moms[i][2]# [X, Y, Z]
        print(f'  cnv:x = {moms[i][0]:5.2f} y = {moms[i][1]:5.2f} d = {moms[i][2]:5.2f}')

    return moms
### -> 座標変換した重心

### 出力画像の表示
def show_images(frame, img_ss, img_depth, img_cls, img_dtc):
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

    fig.suptitle(f"frame {frame}")
    plt.get_current_fig_manager().window.wm_geometry('+500+300')
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
    axTr.set_xlim(-15, 15)
    axTr.set_ylim(-1, 50) #depth
    plt.xlabel('y axis(m)')
    plt.ylabel('depth (m)')

    # データ格納
    dx, dy = [], []
    op_x, op_y = [], []
    count = 0
    id_num = []

    try:
        for ID in range(20): # 全IDに対して
            dx, dy = [], []
            for i in range(len(tracer_wID)):
                for j in range(len(tracer_wID[i])):
                    if ID == tracer_wID[i][j][0]:
                        dx.append(tracer_wID[i][j][1])
                        dy.append(tracer_wID[i][j][3])      # depth

            op_x.append(dx)
            op_y.append(dy)

            color = cm.tab20(ID)
            plt.scatter(op_x[ID], op_y[ID], label=f'id={ID}', color=color, s=5)
            plt.legend()
            # plt.plot(op_x[ID],op_y[ID],color=color)

    except IndexError:
        pass

    ### 重心位置の表示
    plt.show()
    ###

    ### kalman filter
    zs = []
    print(f'zs=')
    for i in range(len(tracer_wID)):
        for j in range(len(tracer_wID[i])):
            if tracer_wID[i][j][0]==targetID_:
                dx = -tracer_wID[i][j][1] # 符号逆
                dy = tracer_wID[i][j][3]
                zs.append(copy.deepcopy([dx,dy]))
                print(f' dx={dx:7.3f}, dy={dy:7.3f}')

    P = np.diag([1,1,1,1]) #位置と速度の共分散行列
    R = np.diag([1, 1])           #観測の共分散行列
    Ms, Ps = run_kf(count=200, R=R, Q=0.01, P=P, zs=zs, do_plot=True)
    ###

    return Ms
###

###
def show_scatter(data):
    windowSize = 8
    figw = plt.figure(figsize=(windowSize,windowSize))
    axW = figw.add_subplot(1,1,1)
    axW.set_title('depth cloud')
    axW.set_xlim(0,640)
    axW.set_ylim(-1,100) #depth
    plt.xlabel('x (pixel)')
    plt.ylabel('depth (m)')

    dx, dy = [], []
    op_x, op_y = [], []

    for nn in data:
        dx.append(nn[0])
        dy.append(nn[1])

    op_x.append(dx)
    op_y.append(dy)

    plt.scatter(op_x, op_y, s=3)
    plt.show()

###

###
def print_tracer(tracer):
    for i in range(len(tracer)):
        for j in range(len(tracer[i])):
            print(f' x={tracer[i][j][0]:6.2f} y={tracer[i][j][1]:6.2f} depth={tracer[i][j][2]:6.2f}')
###
def print_tracerwID(tracer_wID):
    for i in range(len(tracer_wID)):
        for j in range(len(tracer_wID[i])):
            print(f' frame ={i:3d} ID ={tracer_wID[i][j][0]:2.0f} x = {tracer_wID[i][j][1]:7.3f} y = {tracer_wID[i][j][2]:7.3f} depth = {tracer_wID[i][j][3]:.4g}')
###


###
def show_result(tracer_wID, tracer):
    # show_scatter(tracer)
    xs = show_tracer(tracer_wID)

    return xs
###

### 状態 [x dx].T に対する定常速度モデルを実装する KalmanFilter を返す。 """
def pos_vel_filter(x, P, R, Q=0., dt=1.0):

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x[0], x[1], x[2], x[3]]) # 位置と速度
    kf.F = np.array([[1., dt, 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., dt],
                     [0., 0., 0., 1.]])  # 状態遷移行列
    kf.H = np.array([[1., 0, 0, 0],
                     [0, 0, 1., 0]])    # 観測関数
    kf.R *= R                     # 観測の不確実性
    if np.isscalar(P):
        kf.P *= P                 #状態の共分散行列
    else:
        kf.P[:] = P               # [:] を使ってディープコピー

    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf
###

###
def run_kf(x0=(0,0,0,0), P=500, R=0, Q=0, dt=1.0, track=None, zs=None, count=0, do_plot=False, **kwargs):
    print('===run kalman filter===')

    # 初期状態
    x0 = (zs[0][0],0,zs[0][1],-1)

    # カルマンフィルタ作成
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)
    # カルマンフィルタを実行
    xs, cov = [], []
    for z in zs:
        kf.predict()
        kf.update(z)
        xs.append(kf.x)
        cov.append(kf.P)

    xs, cov = np.array(xs), np.array(cov)
    if do_plot:
        op_x, op_y = [], []
        gt_x, gt_y = [], []
        me_x, me_y =[], []
        for xn in xs:
            op_x.append(xn[0])
            op_y.append(xn[2])
        for me in zs:
            me_x.append(me[0])
            me_y.append(me[1])

        windowSize = 8
        figw = plt.figure(figsize=(windowSize,windowSize))
        axW = figw.add_subplot(1,1,1)
        axW.set_xlim(-15,15)
        axW.set_ylim(-1,75) #depth
        plt.xlabel('y axis')
        plt.ylabel('depth')
        # plt.scatter(op_x, op_y, label='kf(xs)',color='b',s=5)
        plt.plot(op_x, op_y)
        plt.scatter(me_x, me_y, label='measurements(zs)',color='y',s=5)
        plt.legend()
        plt.show()

    return xs, cov
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
    maxID_ = 0           # 最大のID
    data = []           # 1frameの重心とID格納
    tracer_wID = []     # IDつきの重心

    a = np.array([0,0])
    b = np.array([0,0])
    distance = 0

    np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
    print(f'\n---all car moment points(tracer)---')
    print_tracer(tracer)

    ### 全フレームに対して処理
    for i in range(len(tracer)):             # 全フレームに対して
        momID = 0
        data = []

        if i==0:    # 最初のフレームに対して
            for j in range(len(tracer[i])):
                momID+=1
                a = np.array(tracer[i][j])
                data.append(np.append(momID,a))
        else:
            print(f' ### estimate ID frame={i}')
            for j in range(len(tracer[i])):     # iの全重心に対して
                a = np.array(tracer[i][j])          # 現在の重心
                print(f'  maxID_={maxID_:2.0f}')

                for k in range(len(tracer_wID[i-1])):   # i-1 の全重心
                    try:
                        b = np.array([tracer_wID[i-1][k][1], tracer_wID[i-1][k][2], tracer_wID[i-1][k][3]])    # 1frame前の重心
                        distance = np.linalg.norm(b-a)
                        np.set_printoptions(precision=3, floatmode='maxprec_equal', suppress=True)
                        print(f'  f={i} a={j} b={k} a={a} b={b} dist={distance:6.3f}')
                        # 距離が閾値以下で同一物体
                        if distance<=tolerance:
                            momID = tracer_wID[i-1][k][0]
                            data.append(np.append(momID,a))
                            if maxID_ < momID: maxID_ = momID
                            print(f'      ^add id:{momID:2.0f}')
                            break
                        # 全重心を探索後同一物体でなければ,新しい物体にID追加
                        if k==len(tracer_wID[i-1])-1 :
                            momID = maxID_ + 1
                            data.append(np.append(momID,a))
                            if maxID_ < momID: maxID_ = momID
                            print(f'      ^new id:{momID:2.0f}')

                    except IndexError as e:
                        print(f'### indexError')
                        pass

        tracer_wID.append(data)     # 1frameの重心データ格納

        print(f'\n---tracer_wID frame={i}---')
        for j in range(len(tracer_wID[i])):
            print(f' ID ={tracer_wID[i][j][0]:2.0f} x = {tracer_wID[i][j][1]:7.3f} y = {tracer_wID[i][j][2]:7.3f} depth = {tracer_wID[i][j][3]:.4g}')
    ###

    return tracer_wID
###

### 一定時間の重心の座標算出
def make_tracer(fnumber, tm_):
    tracer = []
    files = glob.glob(f'{fpath}')
    fname = os.path.basename(files[0]).split('.',1)[0]
    fname = fname[:6] #fname:d10s05
    print(fname)
    ### tm_ frame分実行
    try:

        for ll in range(tm_):
            print(f'\n---estimate frame={ll}---')
            ###ファイル名調整 fnumber:11500
            f1number = '000000'
            f2number = '000000'

            f1  = int(fnumber) + ll*50   #11500+50n
            f1number = f1number[:6-len(str(f1))] + str(f1) #011500
            f2  = int(int(f1)*2/100)  #230+
            f2number = f2number[:6-len(str(f2))] + str(f2) #002300

            gfname  = f'input_zu04a/{fname}_{f1number}.txt'
            ssfname = f'seg_zu04a/{f2number}.png'
            print(f' {gfname}, {ssfname}')
            ###

            ### 重心と画像出力
            moms, img_ss, img_depth, img_cls, img_dtc = estimateMoms(gfname, ssfname)
            img2world(moms) # 座標変換
            tracer.append(moms)
            if ll%number_image_==0: show_images(ll, img_ss, img_depth, img_cls, img_dtc) # n frameごとに
            ###

    except IndexError:
        print('\n===tracer (index error)===')
        print(traceback.format_exc())
        print(*tracer, sep='\n')

    ### id配布
    tracer_wID = decide_mom_id(tracer)
     ## 結果出力
    print('\n=== tracer with ID ===')
    print('frame     |ID    |x      |y      |depth   |')
    for i in range(len(tracer_wID)):
        for j in range(len(tracer_wID[i])):
            print(f'frame ={i:3d} ID ={tracer_wID[i][j][0]:2.0f} x = {tracer_wID[i][j][1]:7.3f} y = {tracer_wID[i][j][2]:7.3f} depth = {tracer_wID[i][j][3]:.4g}')
     ##
    ###

    ### 軌跡表示
    xs = show_result(tracer_wID, tracer)

    print(f'===output===')
    output2csv(xs)

    return 0
###

###
def output2csv(xs):
    print(f'csv path:{csv_path}')
    with open(csv_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerows(xs)
###

### parameter
class_idx_          =  8    # 抽出するクラス 3:human 8:car 10:sign
obj_color_          = 20    # クラスタリングするときの色
min_cluster_size_   = 20    # クラスタリングする点の最少数
tolerance_dc_       =  2    # depthクラスタリングの閾値
tolerance           =  5    # 同一物体と許容する距離 04a11500:3 04a14000sign:5
tolerance_pix_      = 25    # 画像座標で処理するとき
max_range_          = 60    # 考慮する最大距離
tm_                 = 10    # 実行フレーム数
number_image_       =  2    # nフレームごとに画像出力
targetID_           =  1    # 追跡するID
maxID_              =  0    # 最大のID
watcher             = []    # テスト用
fpath = './input_zu04a/d*'
calib_path = './calibration_zu04a/cam_to_cam.yaml'
calib_path_lidar = './calibration_zu04a/cam_to_lidar.yaml'
csv_path = './output_csv/zu04a.csv'
###
def main():
    ### file name
    args = sys.argv
    fnumber = '000000'
    fnumber = args[1]
    timeWidth = args[2]
    ### param
    tm_ = int(timeWidth)
    np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

    ### 処理
    make_tracer(fnumber, tm_)

###
if __name__ == "__main__":
    main()

