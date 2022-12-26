import numpy as np
import scipy as sp
from astropy import constants as c
from astropy import units as u
from scipy.optimize import fsolve
from sympy import Symbol, nsolve, sqrt
from sympy import *
from sympy import symbols
import math
import mpmath
from gekko import GEKKO
from sympy import nsimplify
from sympy import powsimp
from sympy import init_printing
from sympy import sympify
import pyinputplus as pyip

import astropy.units as u
import sympy as sym
units00 = ((1*u.meter).unit.decompose())
U=[unit for unit, power in zip(units00.bases, units00.powers)]
U=(np.append(np.array(U),np.array([u.second,u.kilogram,u.ampere,u.meter,u.Kelvin,u.candela,u.mole])))
L=np.array(np.where(np.array(U) == 'm')).shape    

# Get the number of inputs from the user
num_inputs = int(input("How many inputs do you need (up to 7)? "))

# Initialize an empty array to store the sympy symbols and astropy units
symbols_and_units = []

# Loop up to the number of inputs to get the inputs
for i in range(num_inputs):
  # Get the sympy symbol from the user
  symbol = (input("Enter a sympy symbol: "))

  # Get the astropy unit from the user
  unit = u.Unit(input("Enter an astropy unit: "))

  # Add the symbol and unit to the array
  symbols_and_units.append((symbol, unit))

# Print the array of symbols and units
print(symbols_and_units)



# Initialize a list to store the inputs
x = [0] * 7

# Loop 7 times to get the inputs
for i in range(7):
  # Ask the user different questions for each iteration of the loop
  if i == 0:
    x[i] = float(input("How many lengths? "))
  elif i == 1:
    x[i] = float(input("How many times? "))
  elif i == 2:
    x[i] = float(input("How many masses? "))
  elif i == 3:
    x[i] = float(input("How many currents? "))
  elif i == 4:
    x[i] = float(input("How many temperatures? "))
  elif i == 5:
    x[i] = float(input("How many mols? "))
  else:
    x[i] = float(input("How many luminous intensities? "))

# Print the inputted numbers
print(f"lengths = {x[0]}, times = {x[1]}, masses = {x[2]}, currents = {x[3]}, temperatures = {x[4]}, mols = {x[5]}, luminous intensities = {x[6]}")

ps = x[1]
pkg = x[2]
pA = x[3]
pm = x[0]
pK = x[4]
pcd = x[5]
pmol = x[6]

#Symbolic Solver:
if num_inputs==1:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*u.dimensionless_unscaled
    quantity2=1*u.dimensionless_unscaled
    quantity3=1*u.dimensionless_unscaled
    quantity4=1*u.dimensionless_unscaled
    quantity5=1*u.dimensionless_unscaled
    quantity6=1*u.dimensionless_unscaled
if num_inputs==2:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*(symbols_and_units[1][1])
    quantity2=1*u.dimensionless_unscaled
    quantity3=1*u.dimensionless_unscaled
    quantity4=1*u.dimensionless_unscaled
    quantity5=1*u.dimensionless_unscaled
    quantity6=1*u.dimensionless_unscaled
if num_inputs==3:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*(symbols_and_units[1][1])
    quantity2=1*(symbols_and_units[2][1])
    quantity3=1*u.dimensionless_unscaled
    quantity4=1*u.dimensionless_unscaled
    quantity5=1*u.dimensionless_unscaled
    quantity6=1*u.dimensionless_unscaled
if num_inputs==4:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*(symbols_and_units[1][1])
    quantity2=1*(symbols_and_units[2][1])
    quantity3=1*(symbols_and_units[3][1])
    quantity4=1*u.dimensionless_unscaled
    quantity5=1*u.dimensionless_unscaled
    quantity6=1*u.dimensionless_unscaled
if num_inputs==5:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*(symbols_and_units[1][1])
    quantity2=1*(symbols_and_units[2][1])
    quantity3=1*(symbols_and_units[3][1])
    quantity4=1*(symbols_and_units[4][1])
    quantity5=1*u.dimensionless_unscaled
    quantity6=1*u.dimensionless_unscaled
if num_inputs==6:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*(symbols_and_units[1][1])
    quantity2=1*(symbols_and_units[2][1])
    quantity3=1*(symbols_and_units[3][1])
    quantity4=1*(symbols_and_units[4][1])
    quantity5=1*(symbols_and_units[5][1])
    quantity6=1*u.dimensionless_unscaled
if num_inputs==7:
    quantity0=1*(symbols_and_units[0][1])
    quantity1=1*(symbols_and_units[1][1])
    quantity2=1*(symbols_and_units[2][1])
    quantity3=1*(symbols_and_units[3][1])
    quantity4=1*(symbols_and_units[4][1])
    quantity5=1*(symbols_and_units[5][1])
    quantity6=1*(symbols_and_units[6][1])

quantitymatrix=[quantity0,quantity1,quantity2,quantity3,quantity4,quantity5,quantity6]
if num_inputs==1:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(' ')
    sym2=Symbol(' ')
    sym3=Symbol(' ')
    sym4=Symbol(' ')
    sym5=Symbol(' ')
    sym6=Symbol(' ')
if num_inputs==2:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(symbols_and_units[1][0])
    sym2=Symbol(' ')
    sym3=Symbol(' ')
    sym4=Symbol(' ')
    sym5=Symbol(' ')
    sym6=Symbol(' ')
if num_inputs==3:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(symbols_and_units[1][0])
    sym2=Symbol(symbols_and_units[2][0])
    sym3=Symbol(' ')
    sym4=Symbol(' ')
    sym5=Symbol(' ')
    sym6=Symbol(' ')
if num_inputs==4:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(symbols_and_units[1][0])
    sym2=Symbol(symbols_and_units[2][0])
    sym3=Symbol(symbols_and_units[3][0])
    sym4=Symbol(' ')
    sym5=Symbol(' ')
    sym6=Symbol(' ')
if num_inputs==5:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(symbols_and_units[1][0])
    sym2=Symbol(symbols_and_units[2][0])
    sym3=Symbol(symbols_and_units[3][0])
    sym4=Symbol(symbols_and_units[4][0])
    sym5=Symbol(' ')
    sym6=Symbol(' ')
if num_inputs==6:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(symbols_and_units[1][0])
    sym2=Symbol(symbols_and_units[2][0])
    sym3=Symbol(symbols_and_units[3][0])
    sym4=Symbol(symbols_and_units[4][0])
    sym5=Symbol(symbols_and_units[5][0])
    sym6=Symbol(' ')
if num_inputs==7:
    sym0=Symbol(symbols_and_units[0][0])
    sym1=Symbol(symbols_and_units[1][0])
    sym2=Symbol(symbols_and_units[2][0])
    sym3=Symbol(symbols_and_units[3][0])
    sym4=Symbol(symbols_and_units[4][0])
    sym5=Symbol(symbols_and_units[5][0])
    sym6=Symbol(symbols_and_units[6][0])
    
    

symmatrixint=[sym0,sym1,sym2,sym3,sym4,sym5,sym6]

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

m.options.MAX_ITER=10000
m.options.IMODE=2

cmatrix=[coefficients0,coefficients1,c2,c3,c4,c5,c6]


x0, x1, x2, x3, x4, x5, x6 = [m.Var(1) for i in range(7)]
m.Equations([x0*coefficients0[0]+x1*coefficients1[0] + x2*c2[0] + x3*c3[0] + x4*c4[0] + x5*c5[0] + x6*c6[0] == product[0],\
            x0*coefficients0[1]+x1*coefficients1[1] + x2*c2[1] + x3*c3[1] + x4*c4[1] + x5*c5[1] + x6*c6[1] == product[1],\
            x0*coefficients0[2]+x1*coefficients1[2] + x2*c2[2] + x3*c3[2] + x4*c4[2] + x5*c5[2] + x6*c6[2]== product[2],\
            x0*coefficients0[3]+x1*coefficients1[3] + x2*c2[3]+ x3*c3[3] + x4*c4[3] + x5*c5[3] + x6*c6[3]== product[3],\
            x0*coefficients0[4]+x1*coefficients1[4] + x2*c2[4] + x3*c3[4] + x4*c4[4] + x5*c5[4] + x6*c6[4] == product[4],\
            x0*coefficients0[5]+x1*coefficients1[5] + x2*c2[5] + x3*c3[5] + x4*c4[5] + x5*c5[5] + x6*c6[5] == product[5],\
            x0*coefficients0[6]+x1*coefficients1[6] + x2*c2[6] + x3*c3[6] + x4*c4[6] + x5*c5[6] + x6*c6[6] == product[6]])

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
    scale = (quantity0**xmatrix[0]*quantity1**xmatrix[1]*quantity2**xmatrix[2]*quantity3**xmatrix[3]*quantity4**xmatrix[4]*quantity5**xmatrix[5]*quantity6**xmatrix[6])
xmatrix=[x0.value[0],x1.value[0],x2.value[0],x3.value[0],x4.value[0],x5.value[0],x6.value[0]]
fmatrix=np.where(np.array(xmatrix)==0.0)[0]
if np.where(np.array(xmatrix)==0.0)[0].shape[0]==2:
       xmatrix[fmatrix[0]]=1
       xmatrix[fmatrix[1]]=-1.0

PowersFinal=[power for unit, power in zip(scale.unit.decompose().bases, scale.unit.decompose().powers)]
UnitsFinal=[unit for unit, power in zip(scale.unit.decompose().bases, scale.unit.decompose().powers)]
FinalPowers = np.arange(0,(np.array(PowersFinal).shape[0]))
FinalQuantity = scale.value
for i in range(0, np.array(PowersFinal).shape[0]):
    num=PowersFinal[i]
    roundednum=np.round(num,5)
    
    FinalPowers[i] = roundednum
    FinalQuantity *= UnitsFinal[i]**FinalPowers[i] 


for i in range(0,7):
    if symmatrixint[i]==Symbol(' '):
        symmatrixint[i]=1
        


if xmatrix[0]==1:
        finalsym0=symmatrixint[0]
    #elif x0.value[0]==0.5:
        #finalsym0=sqrt(sym0)
    #elif x0.value[0]==-0.5:
        #finalsym0=1/sqrt(sym0)
else:
    finalsym0=symmatrixint[0]**nsimplify(xmatrix[0])

if xmatrix[1]==1:
    finalsym1=symmatrixint[1]
    #elif x0.value[0]==0.5:
        #finalsym1=sqrt(sym1)
    #elif x0.value[0]==-0.5:
else:
    finalsym1=symmatrixint[1]**nsimplify(xmatrix[1])
if xmatrix[2]==1:
    finalsym2=symmatrixint[2]
else:
    finalsym2=symmatrixint[2]**nsimplify(xmatrix[2])
if xmatrix[3]==1:
    finalsym3=symmatrixint[3]
else:
    finalsym3=symmatrixint[3]**nsimplify(xmatrix[3])
if xmatrix[4]==1:
    finalsym4=symmatrixint[4]
else:
    finalsym4=symmatrixint[4]**nsimplify(xmatrix[4])
if xmatrix[5]==1:
    finalsym5=symmatrixint[5]
else:
    finalsym5=symmatrixint[5]**nsimplify(xmatrix[5])
if xmatrix[6]==1:
    finalsym6=symmatrixint[6]
else:
    finalsym6=symmatrixint[6]**nsimplify(xmatrix[6])


finalsymmatrix=np.array([finalsym0,finalsym1,finalsym2,finalsym3,finalsym4,finalsym5,finalsym6])


if c6==[0,0,0,0,0,0,0]:
    powermatrix = [nsimplify(x0.value[0]),nsimplify(x1.value[0]),nsimplify(x2.value[0]),nsimplify(x3.value[0]),nsimplify(x4.value[0]),nsimplify(x5.value[0])]
    if c5==[0,0,0,0,0,0,0]:
        powermatrix = [nsimplify(x0.value[0]),nsimplify(x1.value[0]),nsimplify(x2.value[0]),nsimplify(x3.value[0]),nsimplify(x4.value[0])]
        if c4==[0,0,0,0,0,0,0]:
            powermatrix = [nsimplify(x0.value[0]),nsimplify(x1.value[0]),nsimplify(x2.value[0]),nsimplify(x3.value[0])]
            if c3==[0,0,0,0,0,0,0]:
                powermatrix = [nsimplify(x0.value[0]),nsimplify(x1.value[0]),nsimplify(x2.value[0])]
                if c2==[0,0,0,0,0,0,0]:
                    powermatrix = [nsimplify(x0.value[0]),nsimplify(x1.value[0])]
                    
FinalThing=(((finalsymmatrix[0]*finalsymmatrix[1]*finalsymmatrix[2]*finalsymmatrix[3]*finalsymmatrix[4]*finalsymmatrix[5]*finalsymmatrix[6])))
if 0.5 in powermatrix:
    FinalThing=sqrt(((finalsymmatrix[0]*finalsymmatrix[1]*finalsymmatrix[2]*finalsymmatrix[3]*finalsymmatrix[4]*finalsymmatrix[5]*finalsymmatrix[6]))**2)
if -0.5 in powermatrix:
    FinalThing=sqrt((finalsymmatrix[0]*finalsymmatrix[1]*finalsymmatrix[2]*finalsymmatrix[3]*finalsymmatrix[4]*finalsymmatrix[5]*finalsymmatrix[6])**2)

pprint(FinalThing)
simplify(FinalThing)