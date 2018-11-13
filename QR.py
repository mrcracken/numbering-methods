# https://github.com/danbar/qr_decomposition
# https://rosettacode.org/wiki/QR_decomposition#Python

# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.eig.html

from math import hypot

import numpy as np


def left_part_matrix(n,a):
    """
    Function for generetic matrix with size n x n
    :param n: size of matrix
    :param a: an element of matrix
    :return: random matrix with size n x n
    """
    buf = np.zeros((n, n))
    flat = buf.ravel()
    flat[0::n+1] = 2
    flat[n::n+1] = -1+a
    flat[1::n+1] = -1-a
    return buf

def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation."""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)

def givens_rotation(A):
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)

if __name__ == '__main__':
    """
    :param n: size of matrix
    :param a: number for left_part_matrix function
    """
    n = 5
    a = 0.5
    
    # Input matrix
    A = left_part_matrix(n,a)
    
    # Print input matrix
    print("Given matrix :\n", A)
    
    # Compute QR decomposition using Givens rotation
    (Q, R) = givens_rotation(A)
    
    # Print orthogonal matrix Q
    print("Q:\n", Q)
    
    # Print upper triangular matrix R
    print("R:\n", R)