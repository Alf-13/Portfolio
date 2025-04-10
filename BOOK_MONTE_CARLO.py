import numpy as np
from scipy import special
import random

k=100 #strike price
t=1 #time to maturity
s=100 #asset price
sig=0.2 #volatility
r=0.06 #continuously compounded rate
div=0.03 #dividend yield
n=10 #time steps
m=1000 #simulations

dt=t/n #time partition
nudt=(r-div-0.5*sig*sig)*dt #risk neutral drift 
sigsdt=sig*(dt**0.5) #volitility rate
lns=np.log(s) #base value

sum_ct=0
sum_ct2=0

for j in range(m):
    lnst=lns
    for i in range(n):
        ee=(2**0.5)*special.erfinv(2*random.random()-1) #random normal z-score
        lnst=lnst+nudt+sigsdt*ee
    st=np.exp(lnst) #reconstitute asset price
    if st>k:
        ct=st-k #value if call is in the money
    else:
        ct=0 #value if call is out of the money
    sum_ct=sum_ct+ct
    sum_ct2=sum_ct2+(ct*ct)

call_value=(sum_ct/m)*np.exp(-r*t)
sd=((sum_ct2-sum_ct*sum_ct/m)*np.exp(-2*r*t)/(m-1))**0.5
se=sd/(m**0.5)

print('Call value: '+str(call_value))

