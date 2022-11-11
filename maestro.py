import numpy as np
import scipy as sp
from astropy import constants as c
from astropy import units as u
from scipy.optimize import fsolve
from sympy import Symbol, nsolve
from sympy import *
import math
import mpmath
from gekko import GEKKO

pi=3.1415926535

astrodict={'c.G', 'c.N_A', 'c.R', 'c.Ryd', 'c.a0', 
           'c.alpha', 'c.b_wien', 'c.c', 'c.e.si', 
           'c.eps0', 'c.g0', 'c.h', 'c.hbar', 'c.k_B',
           'c.m_e', 'c.m_n', 'c.m_p', 'c.mu0', 'c.muB',
           'c.sigma_T', 'c.sigma_sb', 'c.GM_earth', 'c.GM_jup',
           'c.GM_sun', 'c.L_bol0', 'c.L_sun', 'c.M_earth', 'c.M_jup',
           'c.M_sun', 'c.R_earth', 'c.R_jup', 'c.R_sun'}

units00 = ((1*u.meter).unit.decompose())
U=[unit for unit, power in zip(units00.bases, units00.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
L=np.array(np.where(np.array(U) == 'm')).shape



inputvar = np.array([[(c.eps0).value, (c.eps0).unit],[c.mu0.value, c.mu0.unit],[200, u.giganewton]])
#print(c.eps0.unit.decompose())
valsum = 1
unitsum = (u.m/u.m)
for i in range(0, np.shape(inputvar)[0]):
    num = inputvar[i][0]
    unit = inputvar[i][1]
    valsum *= num
    unitsum *= unit

input0=c.hbar
input1=c.c
input2=1*u.dimensionless_unscaled
input3=1*u.dimensionless_unscaled
input4=1*u.dimensionless_unscaled
input5=1*u.dimensionless_unscaled
input6=1*u.dimensionless_unscaled

scalar0=1
scalar1=1
scalar2=1
scalar3=1
scalar4=1
scalar5=1
scalar6=1



print((valsum*unitsum).si)
quantity0=input0*scalar0
quantity1=input1*scalar1
quantity2=input2*scalar2
quantity3=input3*scalar3
quantity4=input4*scalar4
quantity5=input5*scalar5
quantity6=input6*scalar6

ps = -2
pkg = 1
pA = 0
pm = 3
pK = 0
pcd = 0
pmol = 0


units0 = (quantity0.unit.decompose())
P=[power for unit, power in zip(units0.bases, units0.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units0.bases, units0.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
#print(np.array(np.where(np.array(U) == 'm')).shape)
#print(U)

#if np.where(np.array(U) == 's').shape
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0
coefficients0 = [x,y,t,r,b,d,v]
#print(P[m])
#print(coefficients0)

units1 = (quantity1.unit.decompose())
P=[power for unit, power in zip(units1.bases, units1.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units1.bases, units1.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
numb=np.where(np.array(P) == 4)
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0
coefficients1 = [x,y,t,r,b,d,v]
#print(coefficients1)
units2 = (quantity2.unit.decompose())
P=[power for unit, power in zip(units2.bases, units2.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units2.bases, units2.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
numb=np.where(np.array(P) == 4)
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0

c2 = [x,y,t,r,b,d,v]



units3 = (quantity3.unit.decompose())
P=[power for unit, power in zip(units3.bases, units3.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units3.bases, units3.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0
c3 = [x,y,t,r,b,d,v]
#print(c3)


units4 = (quantity4.unit.decompose())
P=[power for unit, power in zip(units4.bases, units4.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units4.bases, units4.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0
c4 = [x,y,t,r,b,d,v]
units5 = (quantity5.unit.decompose())
P=[power for unit, power in zip(units5.bases, units5.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units5.bases, units5.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0
c5 = [x,y,t,r,b,d,v]
units6 = (quantity6.unit.decompose())
P=[power for unit, power in zip(units6.bases, units6.powers)]
P=(np.append(np.array(P),[0,0,0,0,0,0,0]))
U=[unit for unit, power in zip(units6.bases, units6.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
if np.array(np.where(np.array(U) == 's')).shape == L:
    s=np.where(np.array(U) == 's')[0][0]
    x=P[s]
else:
    x=0
if np.array(np.where(np.array(U) == 'kg')).shape == L:
    kg=np.where(np.array(U) == 'kg')[0][0]
    y=P[kg]
else:
    y = 0
if np.array(np.where(np.array(U) == 'A')).shape == L:
    A=np.where(np.array(U) == 'A')[0][0]
    t=P[A]
else:
    t = 0
if np.array(np.where(np.array(U) == 'm')).shape == L:
    m=np.where(np.array(U) == 'm')[0][0]
    r=P[m]
else:
    r = 0
if np.array(np.where(np.array(U) == 'K')).shape == L:
    K=np.where(np.array(U) == 'K')[0][0]
    b=P[K]
else:
    b = 0
if np.array(np.where(np.array(U) == 'cd')).shape == L:
    cd=np.where(np.array(U) == 'cd')[0][0]
    d=P[cd]
else:
    d = 0
if np.array(np.where(np.array(U) == 'mol')).shape == L:
    mol=np.where(np.array(U) == 'mol')[0][0]
    v=P[mol]
else:
    v = 0
c6 = [x,y,t,r,b,d,v]



product = [ps,pkg,pA,pm,pK,pcd,pmol]

m = GEKKO(remote=False)

if quantity0==0 * u.dimensionless_unscaled:
    quantity0=(1* u.dimensionless_unscaled)
else:
    quantity0=quantity0

if quantity1==0 * u.dimensionless_unscaled:
    quantity1=(1* u.dimensionless_unscaled)
else:
    quantity1=quantity1

if quantity2==0 * u.dimensionless_unscaled:
    quantity2=(1* u.dimensionless_unscaled)
else:
    quantity2=quantity2

if quantity3==0 * u.dimensionless_unscaled:
    quantity3=(1* u.dimensionless_unscaled)
else:
    quantity3=quantity3

if quantity4==0 * u.dimensionless_unscaled:
    quantity4=(1* u.dimensionless_unscaled)
else:
    quantity4=quantity4

if quantity5==0 * u.dimensionless_unscaled:
    quantity5=(1* u.dimensionless_unscaled)
else:
    quantity5=quantity5

if quantity6==0 * u.dimensionless_unscaled:
    quantity6=(1* u.dimensionless_unscaled)
else:
    quantity6=quantity6


m.options.MAX_ITER=10000
m.options.IMODE=2

if c6==[0,0,0,0,0,0,0]:
    x0, x1, x2, x3, x4, x5 = [m.Var(1) for i in range(6)]
    m.Equations([x0*coefficients0[0]+x1*coefficients1[0] + x2*c2[0] + x3*c3[0] + x4*c4[0] + x5*c5[0] == product[0],\
            x0*coefficients0[1]+x1*coefficients1[1] + x2*c2[1] + x3*c3[1] + x4*c4[1] + x5*c5[1] == product[1],\
            x0*coefficients0[2]+x1*coefficients1[2] + x2*c2[2] + x3*c3[2] + x4*c4[2] + x5*c5[2] == product[2],\
            x0*coefficients0[3]+x1*coefficients1[3] + x2*c2[3]+ x3*c3[3] + x4*c4[3] + x5*c5[3] == product[3],\
            x0*coefficients0[4]+x1*coefficients1[4] + x2*c2[4] + x3*c3[4] + x4*c4[4] + x5*c5[4] == product[4],\
            x0*coefficients0[5]+x1*coefficients1[5] + x2*c2[5] + x3*c3[5] + x4*c4[5] + x5*c5[5] == product[5],\
            x0*coefficients0[6]+x1*coefficients1[6] + x2*c2[6] + x3*c3[6] + x4*c4[6] + x5*c5[6] == product[6]])
    if c5==[0,0,0,0,0,0,0]:
        x0, x1, x2, x3, x4= [m.Var(1) for i in range(5)]
        m.Equations([x0*coefficients0[0]+x1*coefficients1[0] + x2*c2[0] + x3*c3[0] + x4*c4[0] == product[0],\
        x0*coefficients0[1]+x1*coefficients1[1] + x2*c2[1] + x3*c3[1] + x4*c4[1] == product[1],\
        x0*coefficients0[2]+x1*coefficients1[2] + x2*c2[2] + x3*c3[2] + x4*c4[2] == product[2],\
        x0*coefficients0[3]+x1*coefficients1[3] + x2*c2[3]+ x3*c3[3] + x4*c4[3] == product[3],\
        x0*coefficients0[4]+x1*coefficients1[4] + x2*c2[4] + x3*c3[4] + x4*c4[4] == product[4],\
        x0*coefficients0[5]+x1*coefficients1[5] + x2*c2[5] + x3*c3[5] + x4*c4[5] == product[5],\
        x0*coefficients0[6]+x1*coefficients1[6] + x2*c2[6] + x3*c3[6] + x4*c4[6] == product[6]])
        if c4==[0,0,0,0,0,0,0]:
            x0, x1, x2, x3= [m.Var(1) for i in range(4)]
            m.Equations([x0*coefficients0[0]+x1*coefficients1[0] + x2*c2[0] + x3*c3[0] == product[0],\
            x0*coefficients0[1]+x1*coefficients1[1] + x2*c2[1] + x3*c3[1] == product[1],\
            x0*coefficients0[2]+x1*coefficients1[2] + x2*c2[2] + x3*c3[2]== product[2],\
            x0*coefficients0[3]+x1*coefficients1[3] + x2*c2[3]+ x3*c3[3] == product[3],\
            x0*coefficients0[4]+x1*coefficients1[4] + x2*c2[4] + x3*c3[4] == product[4],\
            x0*coefficients0[5]+x1*coefficients1[5] + x2*c2[5] + x3*c3[5] == product[5],\
            x0*coefficients0[6]+x1*coefficients1[6] + x2*c2[6] + x3*c3[6] == product[6]])
            if c3==[0,0,0,0,0,0,0]:
                x0, x1, x2= [m.Var(1) for i in range(3)]
                m.Equations([x0*coefficients0[0]+x1*coefficients1[0] + x2*c2[0]== product[0],\
                x0*coefficients0[1]+x1*coefficients1[1] + x2*c2[1]== product[1],\
                x0*coefficients0[2]+x1*coefficients1[2] + x2*c2[2]== product[2],\
                x0*coefficients0[3]+x1*coefficients1[3] + x2*c2[3]== product[3],\
                x0*coefficients0[4]+x1*coefficients1[4] + x2*c2[4] == product[4],\
                x0*coefficients0[5]+x1*coefficients1[5] + x2*c2[5] == product[5],\
                x0*coefficients0[6]+x1*coefficients1[6] + x2*c2[6] == product[6]])
else:
        x0, x1, x2, x3, x4, x5, x6 = [m.Var(1) for i in range(7)]
        m.Equations([x0*coefficients0[0]+x1*coefficients1[0] + x2*c2[0] + x3*c3[0] + x4*c4[0] + x5*c5[0] + x6*c6[0] == product[0],\
            x0*coefficients0[1]+x1*coefficients1[1] + x2*c2[1] + x3*c3[1] + x4*c4[1] + x5*c5[1] + x6*c6[1] == product[1],\
            x0*coefficients0[2]+x1*coefficients1[2] + x2*c2[2] + x3*c3[2] + x4*c4[2] + x5*c5[2] + x6*c6[2]== product[2],\
            x0*coefficients0[3]+x1*coefficients1[3] + x2*c2[3]+ x3*c3[3] + x4*c4[3] + x5*c5[3] + x6*c6[3]== product[3],\
            x0*coefficients0[4]+x1*coefficients1[4] + x2*c2[4] + x3*c3[4] + x4*c4[4] + x5*c5[4] + x6*c6[4] == product[4],\
            x0*coefficients0[5]+x1*coefficients1[5] + x2*c2[5] + x3*c3[5] + x4*c4[5] + x5*c5[5] + x6*c6[5] == product[5],\
            x0*coefficients0[6]+x1*coefficients1[6] + x2*c2[6] + x3*c3[6] + x4*c4[6] + x5*c5[6] + x6*c6[6] == product[6]])



m.options.MAX_ITER=1000
m.options.IMODE=2
   

m.solve(disp=False)


if c6==[0,0,0,0,0,0,0]:
    x0.value[0]=np.round(x0.value[0], 10)
    x1.value[0]=np.round(x1.value[0], 10)
    x2.value[0]=np.round(x2.value[0], 10)
    x3.value[0]=np.round(x3.value[0], 10)
    x4.value[0]=np.round(x4.value[0], 10)
    x5.value[0]=np.round(x5.value[0], 10)
    # x(0)==x0.value[0]
    # x(1)==x1.value[0]
    # x(2)==x2.value[0]
    # x(3)==x3.value[0]
    # x(4)==x4.value[0]
    # x(5)==x5.value[0]
    scale = (quantity0**x0.value[0]*quantity1**x1.value[0]*quantity2**x2.value[0]*quantity3**x3.value[0]*quantity4**x4.value[0]*quantity5**x5.value[0]*quantity6)
    if c5==[0,0,0,0,0,0,0]:
        x0.value[0]=np.round(x0.value[0], 10)
        x1.value[0]=np.round(x1.value[0], 10)
        x2.value[0]=np.round(x2.value[0], 10)
        x3.value[0]=np.round(x3.value[0], 10)
        x4.value[0]=np.round(x4.value[0], 10)
        # x(0)==x0.value[0]
        # x(1)==x1.value[0]
        # x(2)==x2.value[0]
        # x(3)==x3.value[0]
        # x(4)==x4.value[0]
        scale = (quantity0**x0.value[0]*quantity1**x1.value[0]*quantity2**x2.value[0]*quantity3**x3.value[0]*quantity4**x4.value[0]*quantity5*quantity6)
        if c4==[0,0,0,0,0,0,0]:
            x0.value[0]=np.round(x0.value[0], 10)
            x1.value[0]=np.round(x1.value[0], 10)
            x2.value[0]=np.round(x2.value[0], 10)
            x3.value[0]=np.round(x3.value[0], 10)
            # x(0)==x0.value[0]
            # x(1)==x1.value[0]
            # x(2)==x2.value[0]
            # x(3)==x3.value[0]
            scale = (quantity0**x0.value[0]*quantity1**x1.value[0]*quantity2**x2.value[0]*quantity3**x3.value[0]*quantity4*quantity5*quantity6)
            if c3==[0,0,0,0,0,0,0]:
                x0.value[0]=np.round(x0.value[0], 10)
                x1.value[0]=np.round(x1.value[0], 10)
                x2.value[0]=np.round(x2.value[0], 10)
                # x(0)==x0.value[0]
                # x(1)==x1.value[0]
                # x(2)==x2.value[0]
                scale = (quantity0**x0.value[0]*quantity1**x1.value[0]*quantity2**x2.value[0]*quantity3*quantity4*quantity5*quantity6)
else:
    x0.value[0]=np.round(x0.value[0], 10)
    x1.value[0]=np.round(x1.value[0], 10)
    x2.value[0]=np.round(x2.value[0], 10)
    x3.value[0]=np.round(x3.value[0], 10)
    x4.value[0]=np.round(x4.value[0], 10)
    x6.value[0]=np.round(x6.value[0], 10)
    x5.value[0]=np.round(x5.value[0], 10)
    # x(0)==x0.value[0]
    # x(1)==x1.value[0]
    # x(2)==x2.value[0]
    # x(3)==x3.value[0]
    # x(4)==x4.value[0]
    # x(5)==x5.value[0]
    # x(6)==x6.value[0]

    scale = (quantity0**x0.value[0]*quantity1**x1.value[0]*quantity2**x2.value[0]*quantity3**x3.value[0]*quantity4**x4.value[0]*quantity5**x5.value[0]*quantity6**x6.value[0])

PowersFinal=[power for unit, power in zip(scale.unit.decompose().bases, scale.unit.decompose().powers)]
UnitsFinal=[unit for unit, power in zip(scale.unit.decompose().bases, scale.unit.decompose().powers)]
FinalPowers = np.arange(0,(np.array(PowersFinal).shape[0]))
FinalQuantity = scale.value
for i in range(0, np.array(PowersFinal).shape[0]):
    num=PowersFinal[i]
    roundednum=np.round(num,5)
    
    FinalPowers[i] = roundednum
    FinalQuantity *= UnitsFinal[i]**FinalPowers[i] 

print(FinalQuantity)
# Var()
# Var(0)==quantity0
# Var(1)==quantity1
# Var(2)==quantity2
# Var(3)==quantity3
# Var(4)==quantity4
# Var(5)==quantity5
# Var(6)==quantity6
check0=0
check1=0
check2=0
check3=0
check4=0
check5=0
check6=0
for i in astrodict:
    if str(input0)==i:
        sym0 = Symbol('-'c.'+ i')
        check0=1
        print(i)


x0.value[0]=sympify(x0.value[0], rational=True)
x1.value[0]=sympify(x1.value[0], rational=True)
x2.value[0]=sympify(x2.value[0], rational=True)
if check0==1:
    if x0.value[0]==1:
        quantity0=scalar0*sym0
    else:
        quantity0=pow(sym0*scalar0,sympify(x0.value[0], rational=True))
if check1==1:
    if x1.value[0]==1:
        quantity1=scalar1*sym1
    else:
        quantity1=pow(sym1*scalar1,sympify(x1.value[0], rational=True))
        

quantity0*quantity1
print(sym0)
for i in astrodict:
    print(i)

    
