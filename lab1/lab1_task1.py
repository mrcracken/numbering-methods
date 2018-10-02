"""
@author: mrcraken
@since: 2018
"""

import numpy as np

def define_matrix(n):
    """
    Function for generetic random matrix with size n x n
    :param n: size of matrix
    :return: random matrix with size n x n
    """
    buf = np.zeros((n, n))
    flat = buf.ravel()
    flat[0::n+1] = 2
    flat[n::n+1] = -1
    flat[1::n+1] = -1
    return buf

def decompose_to_LU(a):
    """
    Decompose matrix of coefficients to L and U matrices.
     L and U triangular matrices will be represented in a single nxn matrix.
    :param a: numpy matrix of coefficients
    :return: numpy LU matrix
    """
    # create emtpy LU-matrix
    lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
    n = a.shape[0]

    for k in range(n):
        # calculate all residual k-row elements
        for j in range(k, n):
            lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        # calculate all residual k-column elemetns
        for i in range(k + 1, n):
            lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

    return lu_matrix

def get_L(m):
    """
    Get triangular L matrix from a single LU-matrix
    :param m: numpy LU-matrix
    :return: numpy triangular L matrix
    """
    L = m.copy()
    for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i+1 :] = 0
    return np.matrix(L)


def get_U(m):
    """
    Get triangular U matrix from a single LU-matrix
    :param m: numpy LU-matrix
    :return: numpy triangular U matrix
    """
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return U

def test_LU(n):
    """
    Function for testing LU-decomposition.
    :param n: is a matrix size
    """
    a = define_matrix(n)
    print('This is matrix a = \n',a)
    LU = decompose_to_LU(a)
    L = get_L(LU)
    U = get_U(LU)
    print('\n Here we multiply L and U \n',L * U)

def solve_LU(lu_matrix, b):
    """
    Solve system of equations from given LU-matrix and vector b of absolute terms.
    :param lu_matrix: numpy LU-matrix
    :param b: numpy matrix of absolute terms [n x 1]
    :return: numpy matrix of answers [n x 1]
    """
    # get supporting vector y
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - lu_matrix[i, :i] * y[:i]

    # get vector of answers x
    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0] )/ lu_matrix[-i, -i]

    return x

def right_part(n):
    """
    Function for generetic random matrix with size n x n
    :param n: size of matrix
    :return: random matrix with size n x n
    """
    buf = np.zeros((n, n))
    flat = buf.ravel()
    flat[0::n+1] = np.random.randint(0, 10)
    flat[n::n+1] = np.random.randint(0, 10)
    flat[1::n+1] = np.random.randint(0, 10)
    return buf

if __name__ == '__main__':
    from numpy import linalg as la
    n = 6
    b = right_part(n)
    lu_matrix = decompose_to_LU(define_matrix(n))
    
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.det.html
    
    Det = la.linalg.det(lu_matrix)
    print("Solve: \n " , solve_LU(lu_matrix, b))
    print("\n Det = " , Det)
