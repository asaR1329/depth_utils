import sys
import re
import copy
from pathlib import Path

args = sys.argv
fname = args[1]
ifname = f'input/{fname}points.txt'
ofname = f'output/{fname}.pcd'
points_num = '0000'
points = 0

n=0
num = 10
dmap = []

# print("---import datalist---")
# print("n = x","c ","r","depth")
with open(ifname) as fp:
    for ln in fp:
        # print ("n =",n,ln.rstrip('\n'))     #情報表示
        data = ln.strip().split(' ')        #1列読み込み
        dmap.append(data);
        n+=1
# print("---end input datalist---",*dmap, sep='\n')
# print("---end input datalist---")

# print("---start output map---")
points_num = f'{n}'
n=0
o_dmap = []
o_data = [0,0,0,0]
path_w = ofname

try:
    with open(path_w,mode='x') as fp:
        print('ldp:make pcd')
except FileExistsError:
    # print('ldp:override pcd')
    pass

scale = 50
with open(path_w, mode='r+') as fp:
    fp.write('VERSION .7\n')
    fp.write('FIELDS x y z rgb\n')
    fp.write(f'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {points_num}\nHEIGHT 1\n')
    fp.write(f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {points_num}\nDATA ascii\n')
    for ln in dmap:

        # for k in range(3):
            # o_data[k] = float(ln[k])
            # if k==2:
                # o_data[k]*=5

        o_data[0] = ( float(ln[2])+7 )/scale * 30
        o_data[1] = (-float(ln[0])+300)/scale
        o_data[2] = (-float(ln[1])+200)/scale

        o_data[3]=(5e+6+float(ln[2])) #RGB8
        o_dmap.append(copy.deepcopy(o_data))
    # print("---end o_map---")

    for ll in o_dmap:
        print(*ll, sep=' ', file=fp) #ファイルに出力
        # print(n+1, *ll, sep=' ')   #確認用
        n+=1

print(f'---end output d_map {fname} {points_num}---')

