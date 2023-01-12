### python3 .py 'number of txt'

import sys
import os
import re
import time
import copy
from pathlib import Path

### txtファイルのデータを格納
def input_depth_points(ifname):
    depthPoints = []
    points_num = 0
    try:
        with open(ifname) as fp:
            for ln in fp:
                data = ln.strip().split(' ')
                depthPoints.append(data);
                points_num += 1
    except FileNotFoundError:
        pass
    return depthPoints, points_num
### ->深度の配列とその数

### PCDを出力
def convert_pcd(fname):
    o_data = [0,0,0,0]          #x,y,z,RGB
    o_dmap = []                 #depthpoints格納
    scale  = 50                 #縮尺
    points_num = 0              #点の数
    points = f'{points_num}'    #点の数

    ifname = f'input/{fname}points.txt'     # 014250points.txt (t=14.25)
    ofname = f'output/{fname}.pcd'

    depthPoints, points_num = input_depth_points(ifname)

    ### PCDファイル作成
    if points_num!=0 and int(fname)%10==0: #点が存在 and 1000/n Hzで出力
        with open(ofname, mode='w') as fp:
            fp.write('VERSION .7\n')
            fp.write('FIELDS x y z rgb\n')
            fp.write(f'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {points_num}\nHEIGHT 1\n')
            fp.write(f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {points_num}\nDATA ascii\n')

            ###txtデータを変換，調整
            for ln in depthPoints:
                o_data[0] = ( float(ln[2])+7  )/scale * 30      #z
                o_data[1] = (-float(ln[0])+300)/scale           #x
                o_data[2] = (-float(ln[1])+200)/scale           #y
                o_data[3] = 10+float(ln[2])*1                   #RGB
                o_dmap.append(copy.deepcopy(o_data))

            for ll in o_dmap:
                print(*ll, sep=' ', file=fp)                    #ファイルに出力

        print(f'---end convert pcd {fname} {points_num}---')    #確認用

    # else:
    #     print('points_num = ', points_num)
### ->pcd file

### txt files => pcd files
def convert_pcds(fname):
    cfname = '000000'   #出力するファイルナンバー

    ### 入力されたファイルまで処理
    for ln in range(int(fname)):
        cfname = '000000'
        ### file name 調整
        cfname = cfname[:6-len(str(ln))] + str(ln)

        ### pcdがない場合に作成する
        # try:
        #     with open(f'output/{cfname}.pcd', mode='x') as fp:
        #         print('convert_txt2pcd:remake pcd file')
        # except FileExistsError:
        #     pass

        convert_pcd(cfname)

        # time.sleep(0.1)

def main():
    args = sys.argv
    fname = args[1]

    convert_pcds(fname)

###
if __name__ == "__main__":
    main()

