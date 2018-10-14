import numpy as np

# https://austingwalters.com/gauss-seidel-method/

def gauss(A, b, x, n):
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.tril.html
    L = np.tril(A)
    U = A - L
    for i in range(n):
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.inv.html
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        print(x)
    return x

A = np.array([[4.0, -2.0, 1.0], [1.0, -3.0, 2.0], [-1.0, 2.0, 6.0]])
b = [1.0, 2.0, 3.0]
x = [1, 1, 1]

n = 5

print (gauss(A, b, x, n))
