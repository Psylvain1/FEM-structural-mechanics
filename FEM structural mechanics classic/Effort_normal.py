import numpy as np


def EN_resistant(Ac,As,fc,fy):
    N= (Ac-As)*fc + As*fy
    return N

def Aire_cerle(Di):
    R=Di/2
    Aire=(R**2)*np.pi
    return Aire
Ac=0.34*0.34 #section de 34 cm par 34 cm
fc= 21 #MPa
As=4*Aire_cerle(0.019)
fy=435 #MPa

N = np.round(EN_resistant(Ac,As,fc,fy)*10**3,0)
print(N)

print(2897/(0.34*0.34))
# 1889 Sans prendre en compte le coffrage
# 2897 en prenant en compte le coffrage
