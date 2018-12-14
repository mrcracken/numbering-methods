"""
@author mrcracken
@since 2018
"""

import numpy as np

def hilb(n, m=0):
    """
    hilb   Hilbert matrix.
       hilb(n,m) is the n-by-m matrix with elements 1/(i+j-1).
       it is a famous example of a badly conditioned matrix.
       cond(hilb(n)) grows like exp(3.5*n).
       hilb(n) is symmetric positive definite, totally positive, and a
       Hankel matrix.
       References:
       M.-D. Choi, Tricks or treats with the Hilbert matrix, Amer. Math.
           Monthly, 90 (1983), pp. 301-312.
       N.J. Higham, Accuracy and Stability of Numerical Algorithms,
           Society for Industrial and Applied Mathematics, Philadelphia, PA,
           USA, 2002; sec. 28.1.
       M. Newman and J. Todd, The evaluation of matrix inversion
           programs, J. Soc. Indust. Appl. Math., 6 (1958), pp. 466-476.
       D.E. Knuth, The Art of Computer Programming,
           Volume 1, Fundamental Algorithms, second edition, Addison-Wesley,
           Reading, Massachusetts, 1973, p. 37.
       NOTE added in porting.  We do not use the function cauchy here to
       generate the Hilbert matrix.  That is done so we can unit test the
       the functions against each other.  Also, the function has been
       generalized to take by row and column sizes.  If only a row size
       is given, we assume a square matrix is desired.
    """
    if n < 1 or m < 0:
        raise ValueError("Matrix size must be one or greater")
    elif n == 1 and (m == 0 or m == 1):
        return np.array([[1]])
    elif m == 0:
        m = n

    return 1. / (np.arange(1, n + 1) + np.arange(0, m)[:, np.newaxis])

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

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration_max(A):
    """
    Compute max eigenvalue and vector
    :param A: input matrix
    :return: max eigenvalue and vector
    """
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new

def power_iteration_min(A):
    """
    Compute min eigenvalue and vector
    :param A: input matrix
    :return: min eigenvalue and vector
    """
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) > 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new

if __name__ == '__main__':
    """
    :param n: dimension of matrix
    """
    n = 5
    
    Hilb = hilb(n)
    Matrix = define_matrix(n)
    
    max_eigenvalue , vector = power_iteration_max(Matrix)
    print("Max eigenvalue = " , max_eigenvalue)
    print("Vector = " , vector)
    print("\n")
    
    min_eigenvalue_hilb , vector_hilb = power_iteration_min(Hilb)
    print("Min eigenvalue = " , min_eigenvalue_hilb)
    print("Vector = " , vector_hilb)
