import numpy as np 
import math

def Theta(t) :
    _theta = t/2 * np.log(t/(2*np.pi)) - t/2 - np.pi/8 + 1/(48*t) - 7/(5760 * np.power(t, 3)) 
    return _theta


Theta(14.13)

def Rt(t, p) :
    return np.power((t/(2*np.pi)), -1/4) * np.cos(2*np.pi*(p * p - p -1/16))/np.cos(2*np.pi * p)

zeta = 0.0
oldzeta = 0.0

def Zt(t) :

    global zeta, oldzeta

    P = math.modf(np.power(t/(2*np.pi), 1/2))[0]
    N = math.modf(np.power(t/(2*np.pi), 1/2))[1]

    Theta_part = 0.0

    for iter in range(1, int(N) + 1):
        Theta_part += 2 * (1/np.power(iter, 1/2) * np.cos(Theta(t) - t*np.log(iter)))
        #print("....T...{:8.5f}".format(Theta_part))
    
    oldzeta = zeta 
    zeta = Theta_part + np.power(-1, N-1) * Rt(t, P)

    if (oldzeta * zeta < 0) : 
        print("{:10.5f}\t{:.3f}\t{:.0f}\t{:8.5f}".format(t, P, N, zeta))

_start = 14
for num in range(0, 20000000) :
    t = _start + num * 0.00001
    Zt(t)