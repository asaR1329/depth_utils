### python3 output_pcd.py <end time> <interval> <start time>

import subprocess
import time
import sys
import os

args = sys.argv
fend = args[1]
finterval = args[2]
fstart = args[3]
timer = '000000'    #ファイル名
ntm  = 10           #スキップするデータ数

for tm in range(int(fend)):
    if tm <= ntm:
        print('nodata')
    else:
        tm *= int(finterval)
        print('tm =',tm)

        try:
            timer = timer[:6-len(str(tm))] + str(tm)

            if int(timer) < int(fstart):
                print('<start time')
            elif os.path.isfile(f'output/{timer}.pcd'):
                cmd = f"rosrun pcl_ros pcd_to_pointcloud output/{timer}.pcd 0.1 _frame_id:=velodyne"   #コマンド入力
                res = subprocess.check_call(cmd.split(), timeout=1.0)
            else:
                print('file not found')

        except subprocess.TimeoutExpired as e:
            print(e)
