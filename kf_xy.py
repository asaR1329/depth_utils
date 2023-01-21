#!/usr/bin/env python
# coding: utf-8

# In[215]:


from filterpy.kalman import predict, update
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import math
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import copy


# In[216]:


def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ 状態 [x dx].T に対する定常速度モデルを実装する KalmanFilter を返す。 """

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


# In[221]:


def compute_obj_data(z_var, process_var, count=1, dt=1.):
    """ objが通った点と観測値からなる2次元 ndarray を返す。 """
    x = np.array([[0.], [-10.]])
    vel = np.array([[1.], [2.]])
    z_std = math.sqrt(np.linalg.det(z_var))
    p_std = math.sqrt(process_var)
    xs = []
    zs = []
    v = [0, 0]
    for tm in range(count):
        v[0] = vel[0] + (randn() * p_std)
        v[1] = vel[1] + (randn() * p_std)
        if tm >= 50:
            v[0] *= 2
            v[1] /= 2
            if tm >= 100:
                v[0] /= -4
                v[1] *= 4
        x[0] += v[0]*dt
        x[1] += v[1]*dt
        
        xs.append(copy.deepcopy([x[0],x[1]]))
        zs.append(copy.deepcopy([x[0] + randn() * z_std, x[1] + randn() * z_std]))

    return np.array(xs), np.array(zs)


# In[224]:


def run(x0=(0.,0.,1.,2.), P=500, R=0, Q=0, dt=1.0,
        track=None, zs=None,
        count=0, do_plot=False, **kwargs):
    """ track はobjの実際の位置、zs は対応する観測値を表す。 """

    # データが与えられないならobjのシミュレーションを実行する。
    if zs is None:
        track, zs = compute_obj_data(R, Q, count)

    # カルマンフィルタを作成する。
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)

    # カルマンフィルタを実行し、結果を保存する。
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
        for gt in track:
            gt_x.append(gt[0])
            gt_y.append(gt[1])
        for me in zs:
            me_x.append(me[0])
            me_y.append(me[1])
            
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.rcParams["figure.figsize"]=[15,10]
        plt.scatter(op_x, op_y, label='kf(xs)',color='b',s=20)
        plt.plot(op_x, op_y)
        plt.scatter(gt_x, gt_y,label='ground truth',color='r',s=5)
        plt.scatter(me_x, me_y, label='measurements(zs)',color='y',s=5)
        plt.legend()
        plt.show()

    return xs, cov


# In[225]:


P = np.diag([500., 49.,50,5]) #位置と速度の共分散行列
R = np.diag([2, 5])        #観測の共分散行列
Ms, Ps = run(count=200, R=R, Q=0.01, P=P, do_plot=True)


# In[220]:


dt = 1.0
x = np.array([10.0,4.5,15.0,3.0])
P = np.diag([500,49,10,10])
F = np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
for xn in range(4):
    x,P = predict(x=x,P=P,F=F,Q=0)
    print('x =', x)
    
H = np.array([[1., 0, 0, 0],[0, 0, 1., 0]])
R = np.diag([5.,4.])
z = np.array([[1.],[2.]])
print(z)
for xn in range(5):
    x,P = update(x,P,z,R,H)
    print('x =',x)


# In[ ]:




