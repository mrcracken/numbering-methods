"""
@author: mrcraken
@since: 2018
"""

from math import sqrt
# https://docs.python.org/2/library/pprint.html
from pprint import pprint
from numpy import linalg as la

def hilbert_matrix(n):
    """
    https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
    Function for define Hilbert matrix
    :param n: size of matrix
    :return: Hilbert matrix with size n x n
    """
    Matrix = [[1/(i+j-1) for i in range(1,n+1)] for j in range(1,n+1)] 
    return Matrix

def isPD(B):
    """
    This function check is matrix is positive-definite
    :param B: input matrix
    :return: true if input matrix is positive-definite, via Cholesky
             false if input matrix is not is positive-definite, via Cholesky
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def cholesky(A):
    """
    Performs a Cholesky decomposition of A, which must 
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L.
    """
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in range(n)]

    try:
        isPD(A)
    # Perform the Cholesky decomposition
        for i in range(n):
            for k in range(i+1):
                tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
                
                if (i == k): # Diagonal elements
                    L[i][k] = sqrt(A[i][i] - tmp_sum)
                else:
                    L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    except Exception as e:
        print("Error")
    return L
 
A = hilbert_matrix(3)
L = cholesky(A)

print("Hilbert matrix:")
pprint(A)

print("Cholesky:")
pprint(L)
