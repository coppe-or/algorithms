#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
    generators, nested_scopes, unicode_literals, with_statement)

"""
A basic example of a Linear Programming Problem solved using PuLP (and NumPy).

In this case, decision variables of types LpInteger or LpContinuous yields the
same results.
"""

import numpy as np
from pulp import *

prob = LpProblem("Giapetto's Problem", LpMaximize)
x = [[LpVariable(str('x_{}_{}').format(i, j), cat=LpContinuous)
        for j in range(2) ]
            for i in range(1)]

c   = [3, 2]
A, b = zip(*[
    ( [2, 1], 100 ),
    ( [1, 1],  80 ),
    ( [1, 0],  40 ),
])

A, b, c, x = map(np.array, (A, b, c, x))

obj = lpSum(c * x)
Ax = A * x

for cn in [ lpSum(Axi)<=bi for Axi, bi in zip(Ax,b) ]:
    prob += cn
prob+= obj

status = prob.solve()

# Printing results:

print(prob)
print('-'*50)

print('status:', status)
print('LpStatus[status]:', LpStatus[status])

optimal_x = tuple(zip(x.ravel(), map(value,x.ravel())))

print('Dec. vars:')
for name, val in optimal_x:
    print("  {}: {}".format(name, val))

print('Objective:', value(prob.objective))

# Example inspired by: http://lipas.uwasa.fi/~tsottine/or_with_octave/or_with_octave.pdf
