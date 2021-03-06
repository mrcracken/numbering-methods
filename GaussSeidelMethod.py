import numpy as np

def gauss(A, b, x, n):
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.tril.html
    L = np.tril(A)
    U = A - L
    for i in range(n):
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.inv.html
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        print(x)
    return x

def unit_test():
    """
    For unit test I took an example from Wikipedia
    https://en.wikipedia.org/wiki/Gauss–Seidel_method
    """
    A = np.array([[16.0, 3.0], [7.0, -11.0]])
    b = [11.0, 13.0]
    x = [1, 1]
    
    n = 5
    
    print (gauss(A, b, x, n))
   
unit_test()
