### python3 output_pcd.py <num_pcd> <interval> <start time>

import subprocess
import time
import sys
import os

args = sys.argv
fnum = args[1]
finterval = args[2]
fstart = args[3]
timer = '000'
for tm in range(int(fnum)):
    if tm <= 5:
        print('nodata')
    else:
        tm *= int(finterval)
        print(tm)
        try:
            if tm < 10:
                timer = '00'+ str(tm)
            elif tm >= 10 and tm < 100:
                timer = '0' + str(tm)
            else:
                timer = str(tm)
            if int(timer) < int(fstart):
                print('skip')
            elif os.path.isfile(f'output/0{timer}.pcd'):
                cmd = f"rosrun pcl_ros pcd_to_pointcloud output/0{timer}.pcd 0.1 _frame_id:=velodyne"
                res = subprocess.check_call(cmd.split(), timeout=0.8)
            else:
                print('skip')
        except subprocess.TimeoutExpired as e:
            print(e)
