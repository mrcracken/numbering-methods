from numpy import array, zeros, diag, diagflat, dot
import numpy as np
import time

def left_part_matrix(n,a):
    """
    Function for generetic matrix with size n x n
    :param n: size of matrix
    :param a: an element of matrix
    :return: random matrix with size n x n
    """
    buf = np.zeros((n, n))
    flat = buf.ravel()
    flat[0::n+1] = a
    flat[n::n+1] = -1-a
    flat[1::n+1] = -1+a
    return buf

def right_part_matrix(n,a):
    # https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
    """
    Function for define right part matrix
    :param n: size of matrix
    :param a: an element of amtrix 
    :return: array with size n
    """
    Array = np.zeros(n)
    Array[0] = 1 - a
    Array[n-1] = 1 + a
    return Array

def jacobi(A,b,N=25,x=None):
    # https://www.quantstart.com/articles/Jacobi-Method-in-Python-and-NumPy
    """
    Solves the equation Ax=b via the Jacobi iterative method
    """
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x

def gauss(A, b, x, n):
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.tril.html
    L = np.tril(A)
    U = A - L
    for i in range(n):
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.inv.html
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        print(x)
    return x

A = left_part_matrix(5,5)
b = right_part_matrix(5,5)
guess = array([1.0,1.0,1.0,1.0,1.0])

start_time = time.time()
sol = jacobi(A,b,N=25,x=guess)

print ("A: \n" , A)

print ("b: \n" , b)

print ("x: \n" , sol)

print("--- %s seconds ---" % (time.time() - start_time))
