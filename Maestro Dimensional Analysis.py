#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# Function to parse a mathematical expression and return a list of symbols
def parse_expression(expression):
  # Initialize an empty set to store the symbols in the expression
  symbols = set()

  # Loop over each character in the expression
  for char in expression:
    # If the character is an alphabetical letter, add it to the set of symbols
    if char.isalpha():
      symbols.add(char)

  # Return the list of symbols
  return list(symbols)

import sympy as sym

def assign_units(expression):
  # Create a dictionary to store the units of each symbol
  symbol_units = {}

  # Replace all instances of ^ with **
  expression = expression.replace("^", "**")

  # Parse the expression to get a list of symbols
  symbols = parse_expression(expression)

  # Ask the user to input the units for each symbol
  for symbol in symbols:
    symbol_units[symbol] = input(f"Enter the units for {symbol}: ")

  # Use the astropy unit parser to convert the inputted units to astropy units
  for symbol, unit in symbol_units.items():
    symbol_units[symbol] = u.Unit(unit).decompose()

  # Evaluate the expression with the assigned units
  result = eval(expression, symbol_units)

  # Print the final expression using sympy
  pprint(f"The expression's units are: {(result)}")

while True:
  # Get the inputted expression from the user
  expression = input("Enter a symbolic mathematical expression: ")

  # Assign units to the symbols in the expression and output the final expression
  assign_units(expression)

  # Ask the user if they want to compute another quantity
  another = input("Do you want to compute another quantity? (yes/no) ")

  # If the user does not want to compute another quantity, exit the loop
  if another.lower() == "no":
    break

