import numpy as np
from fractions import Fraction
from sympy import *
from sympy import symbols
from sympy.physics.units import *
from sympy.physics.units.systems import SI
from sympy.physics.units import length, mass, acceleration, force
from sympy.physics.units.systems.si import *
#L, T, M, K, Q, N, C = symbols('L T M K Q N C')
#m,s,H,F,C,A,eV,ohm,V,J, W, mol, P, kg   = symbols('m s H F C A eV ohm V J W mol P kg')
def find_dims(exprinitial):
    m,s,H,F,C,A,eV,ohm,V,J, W, mol, P, kg, newt   = symbols('m s H F C A eV ohm V J W mol P kg newt')
    L, T, M, K, Q, N, C = symbols('L T M K Q N C')
    exprinitial = sympify(exprinitial)
    exprdim = exprinitial.subs([(m,length),(kg,mass), (s,time), (H, inductance), (F,capacitance), (C, charge), 
        (A,current), (J,energy),(eV,energy), (K, temperature), (ohm, voltage/current),(V,volt),(T, magnetic_flux/length**2), 
        (c, luminous_intensity), (newt, force),(mol, amount_of_substance), (W, power), (P, pressure)])
    systemdims = exprdim
    dimsys_default
    dimsys_default.get_dimensional_dependencies(systemdims)
    dimmatrix = np.array([])
    dimfinal = []
    for dims in dimsys_default.get_dimensional_dependencies(systemdims):
        dim, powers = dims, dimsys_default.get_dimensional_dependencies(systemdims)[dims]
        dimmatrix = np.vstack((dim, powers))
        dimfinal.append(dimmatrix)
    expressionfin = 1
    dimdict = {length:L, time:T, mass:M, current:Q/T, temperature:K, amount_of_substance:N, luminous_intensity:C}
    for i in range (0,np.shape(dimfinal)[0]):
        expressionfin = expressionfin*(dimdict[dimfinal[i][0][0]]**dimfinal[i][1][0])

    return expressionfin
# Function to validate and convert a string to a SymPy expression
def parse_expression(expression_str):
    try:
        expression_str = expression_str.replace("N", "newt")
        expr = sympify(expression_str)
        return expr
    except SympifyError:
        print("Invalid expression. Please try again.")
        return None

# Input: Number of expressions to input
numb = int(input("Enter the number of expressions (n): "))

expressions = []
symbollist = []

for _ in range(numb):
    while True:
        symbol = input("Enter a symbol to represent the expression: ")
        expression_str = input("Enter a mathematical expression: ")
        parsed_expr = parse_expression(expression_str)
        parsed_expr = find_dims(parsed_expr)
        if parsed_expr is not None and type(eval(expression_str)) is not float and type(eval(expression_str)) is not int:
            expressions.append(parsed_expr)
            symbollist.append(Symbol(symbol))
            break
        else:
            print("Invalid. Please enter an expression with dimensions: ")

# Prompt the user to input a final mathematical expression
rexpression_str = input("\nEnter a final mathematical expression: ")
rexpression = find_dims(sympify(parse_expression(rexpression_str)))

# Print the parsed expressions
print("Dimensions Key: L:Length, T:Time, M:Mass, K=Temperature, Q:Charge, N:Amount of Substance, C:Luminous Intensity")
print("\nParsed Expressions:")
for i, (expr, symbol) in enumerate(zip(expressions, symbollist), start=1):
    print(f"Expression {i}: [{symbol}] = [{expr}]")


L, T, M, K, Q, N, C = symbols('L T M K Q N C')

# Assuming expressions is a list of expressions
  # Define your list of expressions here

# Create an empty 2D array for columns
columns = np.empty((0, 7))

for i in range(len(expressions)):
    if expressions[i]!=1:
        q = expressions[i]
        Lvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (Q, 1), (N, 1), (C, 1)]), L).subs(L, E))
        Tvec = float(log(q.subs([(L, 1), (M, 1), (K, 1), (Q, 1), (N, 1), (C, 1)]), T).subs(T, E))
        Mvec = float(log(q.subs([(L, 1), (T, 1), (K, 1), (Q, 1), (N, 1), (C, 1)]), M).subs(M, E))
        Ovec = float(log(q.subs([(M, 1), (T, 1), (L, 1), (Q, 1), (N, 1), (C, 1)]), K).subs(K, E))
        Cvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (L, 1), (N, 1), (C, 1)]), Q).subs(Q, E))
        Nvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (Q, 1), (L, 1), (C, 1)]), N).subs(N, E))
        Hvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (Q, 1), (N, 1), (L, 1)]), C).subs(C, E))
        
        vector = np.array([Lvec, Tvec, Mvec, Ovec, Cvec, Nvec, Hvec])
    if expressions[i]==1:
        vector = np.array([0, 0, 0, 0, 0, 0, 0])
    columns = np.vstack((columns, vector))

M1=np.transpose(columns)
    
print(M1)
if rexpression!=1:
    q = rexpression
    Lvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (Q, 1), (N, 1), (C, 1)]), L).subs(L, E))
    Tvec = float(log(q.subs([(L, 1), (M, 1), (K, 1), (Q, 1), (N, 1), (C, 1)]), T).subs(T, E))
    Mvec = float(log(q.subs([(L, 1), (T, 1), (K, 1), (Q, 1), (N, 1), (C, 1)]), M).subs(M, E))
    Ovec = float(log(q.subs([(M, 1), (T, 1), (L, 1), (Q, 1), (N, 1), (C, 1)]), K).subs(K, E))
    Cvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (L, 1), (N, 1), (C, 1)]), Q).subs(Q, E))
    Nvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (Q, 1), (L, 1), (C, 1)]), N).subs(N, E))
    Hvec = float(log(q.subs([(M, 1), (T, 1), (K, 1), (Q, 1), (N, 1), (L, 1)]), C).subs(C, E))
      
    solvector = np.array([Lvec, Tvec, Mvec, Ovec, Cvec, Nvec, Hvec])
if rexpression==1:
    solvector = np.array([0, 0, 0, 0, 0, 0, 0])
print('Solution Vector:', solvector)
# Create the dimension matrix M
#M = np.array([[1, 1, 0, 0],
#              [2, 2, 0, 0],
#              [-1, -2, -1, 0],
#              [0, -1, 0, 1]])
# Solve for the vector v
if rexpression==1:
    _, _, V = np.linalg.svd(M1)
    v = V[-1]
    v_normalized = v / v[0]
if rexpression !=1:
# Define your matrix M and vector u
    M = M1  # Replace with your matrix
    u = solvector                # Replace with your vector

    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    M_pinv = np.dot(Vh.T, np.dot(np.diag(1 / s), U.T))
    v = np.dot(M_pinv, u)
    v_normalized = v
# Round the final numbers in v_normalized to be rationals
v_rounded = [Fraction(num).limit_denominator() for num in v_normalized]

print("Dimension Matrix M:")
print(M1)
print("\nSolution vector v (normalized and rounded):")
print(v_normalized)

#symbolarr = [h,k,w,T]
symbolarr = symbollist
calcarr = []
for i in range (0, np.shape(np.array(symbolarr))[0]):
    calcappender = symbolarr[i]**nsimplify(v_normalized[i])
    calcarr = np.append(calcappender, calcarr)
expr = 1
for i in range (0, np.shape(np.array(symbolarr))[0]):
    expr = expr*calcarr[i]


print('Dimensions of Characteristic Expression: ', rexpression)
pprint('Characteristic Expression: ')
pprint(expr)
