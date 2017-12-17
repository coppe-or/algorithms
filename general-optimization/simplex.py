#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
    generators, nested_scopes, unicode_literals, with_statement)

"""
A basic example of a Linear Programming Problem solved using the Simplex Method
"""

# TODO: Complete and test the method for cases other than the single optimal solution

__all__ = ('SimplexTableau',)

import numpy as np

# Simplex Tableau Example (canonical form)
TablExample = np.array(
[   # X1 X2  S1 S2 S3     b
    [  2, 1,  1, 0, 0,  100 ], #  2*X1 + 1*X2 <= 100
    [  1, 1,  0, 1, 0,   80 ], #  1*X1 + 1*X2 <=  80
    [  1, 0,  0, 0, 1,   40 ], #  1*X1 + 0*X2 <=  40
    [ -3,-2,  0, 0, 0,    0 ], #  obj: Max(3*X1+2*X2)
], dtype=np.float)

_diagram_ = ("""
         +-------------------+-----+
  matrix | X1... S1... R1... | RHS |
+--------+-------------------+-----+
|  Base  |         A         |  b  |
+--------+-------------------+-----+
|    P   |        c*X        | obj |
+--------+-------------------+-----+\
""")

def SimplexTableau(Tabl, s='m', r=0):
    """Simplex Tableau:
    Shows the steps until find best feasible solution if any.
        args:
            Tabl: tableau array in canonical form
            s:    number of slack variables (default: m)
            r:    number of artificial variables (default: 0)
    """
    Tabl = np.array(Tabl)
    Ab = Tabl[:-1]
    A, RHS = Ab[:,:-1], Ab[:,-1]
    P = Tabl[-1]
    P_ = P[:-1]
    m, n = A.shape
    s = int(s) if s!='m' else m
    n = n-s-r
    # Column names: X, S, R for normal, slack and artificial
    col_names = np.array(tuple(
        var_name_by_j(j, n, s)
            for j in range(n+s+r)) + ('RHS',), dtype=np.str)

    print(
    """Simplex Tableau:
    {_diagram_}
    \ncol_names:\n{col_names}
    \nA:\n{A}
    \nP:\n{P}
    \nRHS: {RHS}.T
    \nm, n, s, r:\n{m}, {n}, {s}, {r}
    """.format(_diagram_=_diagram_,**locals()))

    step=0
    print('# step: %s' % (step,))
    print(tableau_str(Ab, P, col_names))
    while (P<0).tolist().count(True)>0:
        step+=1
        print('-'*50)
        piv = find_pivot(P, Ab, RHS)
        i, j = piv
        print('pivot:', (i+1,j+1))
        pivot_tabl(piv, P, Ab)
        print('-'*50)
        print('# step: %s' % (step,))
        print(tableau_str(Ab, P, col_names))

def arg_min_pos(arr):
    """Returns the index of the (first occurrence of the) minimum positive
    value of the array
    """
    a = np.array(arr)
    idx = np.argwhere(a>0).ravel()
    return idx[np.argmin(a[idx])]

def arg_max_pos(arr):
    """Returns the index of the (first occurrence of the) maximum positive
    value of the array
    """
    a = np.array(arr)
    idx = np.argwhere(a>0).ravel()
    return idx[np.argmax(a[idx])]

def var_name_by_j(j, n, s, start=1):
    """
    args:
        j: (0-based) index of column
        n: number of primary decision variables
        s: number of slack variables
        start: start of subscript index for printing
    """
    if j<n:   return 'X%s' % (start+j,)
    if j<n+s: return 'S%s' % (start+j-n,)
    return           'R%s' % (start+j-n-s,)

def find_pivot(P, Ab, RHS):
    j = P[:-1].argmin()
    A_j = Ab[:,j]
    idx = np.argwhere(A_j>0).ravel()
    ratio_test = (RHS[idx]/A_j[idx])
    i = idx[arg_min_pos(ratio_test)]
    fmt = lambda args: '%6s/%-3s = %s'% args
    print(
        'Ratio test (j=%s):'%(j+1,),
        '  i  RHS/Aij',
        '\n'.join( '%3s  %s'%(i+1, v)
            for i, v in zip(
                idx, map(fmt, zip(RHS[idx], A_j[idx], ratio_test)))),
        sep='\n')

    return (i, j)

def pivot_tabl(piv, P, Ab):
    i, j = piv
    Ab[i,:] /= Ab[piv]
    piv_row = Ab[i,:]
    for k, akj in enumerate(Ab[:,j]):
        if i!=k:
            Ab[k,:] -= Ab[k,j] * piv_row
    P[:] -= P[j] * piv_row

def find_basis(A):
    "return an array of pairs (i,j) where whe find the ones in basis columns"
    A = np.array(A)
    m = len(A)
    pairs =_find_basis(A)
    return pairs[:m]

def _find_basis(A):
    return np.array([
        (i, j) for i,j in np.argwhere(A==1)
            if np.sum(A[:,j])==1 ])

def tab_str(tab, row_sep='\n', col_sep=' '):
    "right-aligned printable str table of 'tab' contents"
    tab = np.array(tab)
    cells = np.ravel(tab.astype(np.str))
    Len = max(map(len,cells))
    fmt = '{:>%ss}' % (Len,)
    cells = np.array(tuple(map(fmt.format, cells)))
    cells = cells.reshape(*tab.shape)
    return  row_sep.join(map(col_sep.join, cells))

def tableau_str(Ab, P, col_names):
    basis = find_basis(Ab[:,:-1])
    rows, cols = basis.T

    row_names = col_names[cols].astype(np.str).tolist()
    row_names.insert(0, '')
    row_names.append('P')
    row_names = np.array([row_names]).T

    tableau_vals = np.vstack((
        [col_names],
        Ab,
        P,
    ))

    tableau_vals = np.hstack((row_names, tableau_vals))

    return '{}'.format(tab_str(tableau_vals))

if __name__ == '__main__':
    SimplexTableau(TablExample)

