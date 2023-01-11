### python3 txt2pcd.py 'number txtfile'

import subprocess
import time
import sys
import os

args = sys.argv
numf = args[1]
fname = '000'
path = 'aaa.txt'

for tm in range(int(numf)):
    # print('t2p:tm =', tm)
    if tm < 10:
        fname = '00'+str(tm)
    elif tm >= 10 and tm < 100:
        fname = '0'+str(tm)
    else:
        fname = str(tm)
    path = f'input/0{fname}points.txt'
    try:
        cmd = f'python3 load_depth_points.py 0{fname}'
        res = subprocess.check_call(cmd.split())
    except :
        print('t2p:error')

