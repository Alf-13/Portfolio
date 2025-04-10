import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from datetime import date

file_path='C:/Users/alfre/Desktop/PYTHON FILES/MORNING POST/'
symbol=input('INPUT STOCK TICKER:')
df=pd.read_csv(f'{file_path}{date.today()}/price charts/{symbol}.csv')
top_bound=pd.DataFrame(columns=['position','price'])
bottom_bound=pd.DataFrame(columns=['position','price'])
top_bound_critical=pd.DataFrame(columns=['position','price'])
bottom_bound_critical=pd.DataFrame(columns=['position','price'])
L1=250
sr=1000
count=0
price=pd.DataFrame(df["4. close"].iloc[0:L1])
price=price[::-1].reset_index(drop=True)
p_avg=price['4. close'].mean()
price=price.to_numpy()
t=np.arange(0,L1,1)

plt.style.use('seaborn-poster')

form=price[:,0]

L=len(t)
#
inc=7*L/sr
#
holding_arr=np.zeros(((L-2),10))
for x in range(1,(L-1)):
    flag_tail=0
    flag_head=0
    up_theta_tail=0
    up_mag_tail=0
    down_theta_tail=0
    down_mag_tail=0
    up_theta_head=0
    up_mag_head=0
    down_theta_head=0
    down_mag_head=0
    point=[t[x],form[x]]
    for y in range(x-1):
        t_dif_tail=t[y]-point[0]
        form_dif_tail=form[y]-point[1]
        mag_tail=(t_dif_tail*t_dif_tail+form_dif_tail*form_dif_tail)**0.5
        theta_tail=np.arctan(form_dif_tail/t_dif_tail)+np.pi
        if flag_tail==0:
            up_theta_tail=theta_tail
            down_theta_tail=theta_tail
            up_mag_tail=mag_tail
            down_mag_tail=mag_tail
            up_wall_tail=np.sin(theta_tail)
            down_wall_tail=np.sin(theta_tail)
            flag_tail=1
        else:
            wall_tail=np.sin(theta_tail)
            if wall_tail>up_wall_tail:
                up_wall_tail=wall_tail
                up_theta_tail=theta_tail
                up_mag_tail=mag_tail                
            if wall_tail<down_wall_tail:
                down_wall_tail=wall_tail
                down_theta_tail=theta_tail
                down_mag_tail=mag_tail
    for z in range((x+1),L):
        t_dif_head=t[z]-point[0]
        form_dif_head=form[z]-point[1]
        mag_head=(t_dif_head*t_dif_head+form_dif_head*form_dif_head)**0.5
        theta_head=np.arctan(form_dif_head/t_dif_head)
        if flag_head==0:
            up_theta_head=theta_head
            down_theta_head=theta_head
            up_mag_head=mag_head
            down_mag_head=mag_head
            up_wall_head=np.sin(theta_head)
            down_wall_head=np.sin(theta_head)
            flag_head=1
        else:
            wall_head=np.sin(theta_head)
            if wall_head>up_wall_head:
                up_wall_head=wall_head
                up_theta_head=theta_head
                up_mag_head=mag_head
            if wall_head<down_wall_head:
                down_wall_head=wall_head
                down_theta_head=theta_head
                down_mag_head=mag_head
    if abs(up_mag_tail*np.cos(up_theta_tail))>inc:
        up_tail=[(point[0]+up_mag_tail*np.cos(up_theta_tail)),(point[1]+up_mag_tail*np.sin(up_theta_tail))]
    else:
        up_tail=[0,p_avg]
        count=count+1
    if abs(down_mag_tail*np.cos(down_theta_tail))>inc:
        down_tail=[(point[0]+down_mag_tail*np.cos(down_theta_tail)),(point[1]+down_mag_tail*np.sin(down_theta_tail))]
    else:
        down_tail=[0,p_avg]
        count=count+1
    if abs(up_mag_head*np.cos(up_theta_head))>inc:
        up_head=[(point[0]+up_mag_head*np.cos(up_theta_head)),(point[1]+up_mag_head*np.sin(up_theta_head))]
    else:
        up_head=[0,p_avg]
        count=count+1
    if abs(down_mag_head*np.cos(down_theta_head))>inc:
        down_head=[(point[0]+down_mag_head*np.cos(down_theta_head)),(point[1]+down_mag_head*np.sin(down_theta_head))]
    else:
        down_head=[0,p_avg]
        count=count+1
    pt=point[0]
    pf=point[1]
    utt=up_tail[0]
    uft=up_tail[1]
    uth=up_head[0]
    ufh=up_head[1]
    dtt=down_tail[0]
    dft=down_tail[1]
    dth=down_head[0]
    dfh=down_head[1]
    inserter=[pt,pf,utt,uft,dtt,dft,uth,ufh,dth,dfh]
    holding_arr[(x-1),:]=inserter
critical_points=pd.DataFrame(holding_arr,columns=['point_t','point_form','up_t_tail','up_form_tail','down_t_tail','down_form_tail','up_t_head','up_form_head','down_t_head','down_form_head'])
for i in range(len(critical_points)):
    top_bound.loc[len(top_bound)]=[int(round(critical_points['up_t_tail'].iloc[i])),critical_points['up_form_tail'].iloc[i]]
    top_bound.loc[len(top_bound)]=[int(round(critical_points['up_t_head'].iloc[i])),critical_points['up_form_head'].iloc[i]]
    bottom_bound.loc[len(bottom_bound)]=[int(round(critical_points['down_t_tail'].iloc[i])),critical_points['down_form_tail'].iloc[i]]
    bottom_bound.loc[len(bottom_bound)]=[int(round(critical_points['down_t_head'].iloc[i])),critical_points['down_form_head'].iloc[i]]
top_bound=top_bound.drop_duplicates(subset='position')
bottom_bound=bottom_bound.drop_duplicates(subset='position')
top_bound=top_bound.sort_values(by=['position'],ascending=True,ignore_index=True)
bottom_bound=bottom_bound.sort_values(by=['position'],ascending=True,ignore_index=True)
top_bound=top_bound.drop(labels=0,axis=0)
bottom_bound=bottom_bound.drop(labels=0,axis=0)
top_bound=top_bound.to_numpy()
bottom_bound=bottom_bound.to_numpy()

day_new=np.linspace(0,L1,sr)
f_top=CubicSpline(top_bound[:,0],top_bound[:,1],bc_type='natural')
top_bound_new=f_top(day_new)
f_bottom=CubicSpline(bottom_bound[:,0],bottom_bound[:,1],bc_type='natural')
bottom_bound_new=f_bottom(day_new)

top_bound_critical.loc[len(top_bound_critical)]=[t[0],form[0]]
bottom_bound_critical.loc[len(bottom_bound_critical)]=[t[0],form[0]]
for j in range(sr-3):
    if (top_bound_new[j+1]-top_bound_new[j])>=0 and (top_bound_new[j+2]-top_bound_new[j+1])<=0:
       top_bound_critical.loc[len(top_bound_critical)]=[day_new[j+1],top_bound_new[j+1]]
    if (bottom_bound_new[j+1]-bottom_bound_new[j])<=0 and (bottom_bound_new[j+2]-bottom_bound_new[j+1])>=0:
       bottom_bound_critical.loc[len(bottom_bound_critical)]=[day_new[j+1],bottom_bound_new[j+1]]
top_bound_critical.loc[len(top_bound_critical)]=[t[L1-1],form[L1-1]]
bottom_bound_critical.loc[len(bottom_bound_critical)]=[t[L1-1],form[L1-1]]
top_bound_critical=top_bound_critical.to_numpy()
bottom_bound_critical=bottom_bound_critical.to_numpy()

f_top_critical=CubicSpline(top_bound_critical[:,0],top_bound_critical[:,1],bc_type='natural')
top_bound_critical_new=f_top_critical(day_new)
f_bottom_critical=CubicSpline(bottom_bound_critical[:,0],bottom_bound_critical[:,1],bc_type='natural')
bottom_bound_critical_new=f_bottom_critical(day_new)

trend=(top_bound_critical_new+bottom_bound_critical_new)/2

up_tail_arr=[critical_points['up_t_tail'].iloc[:],critical_points['up_form_tail'].iloc[:]]
down_tail_arr=[critical_points['down_t_tail'].iloc[:],critical_points['down_form_tail'].iloc[:]]
up_head_arr=[critical_points['up_t_head'].iloc[:],critical_points['up_form_head'].iloc[:]]
down_head_arr=[critical_points['down_t_head'].iloc[:],critical_points['down_form_head'].iloc[:]]

print('Adjacent Point Ratio: '+str(count/(4*len(critical_points))))

plt.plot(t,form,'r')
#plt.plot(up_tail_arr[0],up_tail_arr[1],'.')
plt.plot(down_tail_arr[0],down_tail_arr[1],'.')
#plt.plot(up_head_arr[0],up_head_arr[1],'.')
plt.plot(down_head_arr[0],down_head_arr[1],'.')
#plt.plot(day_new,top_bound_new,'b')
#plt.plot(day_new,bottom_bound_new,'b')
plt.plot(day_new,top_bound_critical_new,'g')
plt.plot(day_new,bottom_bound_critical_new,'g')
plt.plot(day_new,trend,'k')
#plt.legend(['function','up_tail','down_tail','up_head','down_head','top_bound','bottom_bound','top_bound_critical','bottom_bound_critical'],loc='upper right')
plt.show()

print('Program Complete')




















