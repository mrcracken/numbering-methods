from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import time

def jacobi(A,b,N=25,x=None):
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

def unit_test():
    """
    For our unit test I took an example from Wikipedia
    https://en.wikipedia.org/wiki/Jacobi_method
    """
    A = array([[2.0,1.0],[5.0,7.0]])
    b = array([11.0,13.0])

    start_time = time.time()
    sol = jacobi(A,b,N=25,x=None)

    print ("A:")
    pprint(A)

    print ("b:")
    pprint(b)

    print ("x:")
    pprint(sol)

    print("--- %s seconds ---" % (time.time() - start_time))
    
unit_test()
