### python3 load_depth_points.py 'input file path'

import sys
import re
import copy
from pathlib import Path

args = sys.argv
fpath = args[1]
fname = 'output'
ifname = f'{fpath}'
ofpath = 'output_pcd'
ofname = f'{ofpath}/{fname}.pcd'

n=0
num = 10
dmap = []

with open(ifname) as fp:
    for ln in fp:
        data = ln.strip().split(' ')        #1列読み込み
        dmap.append(data);
        n+=1

points_num = f'{n}'
n=0
o_dmap = []
o_data = [0,0,0,0]
path_w = ofname

try:
    with open(path_w,mode='x') as fp:
        print('ldp:make pcd')
except FileExistsError:
    pass

### データ出力 ###
scale = 50
with open(path_w, mode='r+') as fp:
    fp.write('VERSION .7\n')
    fp.write('FIELDS x y z rgb\n')
    fp.write(f'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {points_num}\nHEIGHT 1\n')
    fp.write(f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {points_num}\nDATA ascii\n')
    for ln in dmap:

        o_data[0] = ( float(ln[2])+7 )/scale * 30   #z
        o_data[1] = (-float(ln[0])+300)/scale       #x
        o_data[2] = (-float(ln[1])+200)/scale       #y

        o_data[3] = float(ln[2]) #RGB8
        o_dmap.append(copy.deepcopy(o_data))

    for ll in o_dmap:
        print(*ll, sep=' ', file=fp) #ファイルに出力
        n+=1

print(f'---end output d_map {fname} {points_num}---')

