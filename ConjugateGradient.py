"""
@author: mrcraken
@since: 2018
"""

import numpy as np
import matplotlib.pyplot as plt

def hilb(n, m=0):
    # https://gist.github.com/fabianp/1046959
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

    """
    :param A: input matrix A
    :param b: right part matrix of Ax=b
    :param tol: accuracy
    :it_max: max iteration steps
    :return: solution x and number of iteration steps
    """
def cg(A, b, tol, it_max):
	it=0; x = 0;
	r = np.copy(b); r_prev = np.copy(b)
	rho = np.dot(r,r)
	p = np.copy(r)
	while (np.sqrt(rho) > tol*np.sqrt(np.dot(b,b)) and it < it_max):
		it += 1
		if it == 1:
			p[:] = r[:]
		else:
			beta = np.dot(r,r)/np.dot(r_prev,r_prev)
			p = r + beta*p
		w = np.dot(A, p)
		alpha = np.dot(r,r)/np.dot(p, w)
		x = x + alpha*p
		r_prev[:] = r[:]
		r = r - alpha*w
		rho = np.dot(r,r)
	return x, it
    
def unit_test(n, tol, it_max):
    A = hilb(n)
    b = A.sum(axis=1)
    sol , it = cg(A, b, tol, it_max)
    return sol, it
    
def plot(m):
    """
    Make a plot
    :param m: max iteration steps
    """
    tol = 0.000001
    it_max = 25
    it = np.zeros(shape=(m))
    n = np.zeros(shape=(m))
    for i in range (1,m):
            sol, it[i] = unit_test(i, tol, it_max)
            n[i] = i
    plt.plot(it, n)
    plt.xlabel('Iterations')
    plt.ylabel('n')
    plt.show()
        
    
if __name__ == "__main__":
    # matrix size for function cg(n, tol, it_max)
    n = 5
    # max matrix size for function plot(m)
    m = 20
    # accuracy for function cg()
    tol = 0.000001
    # max iteration steps for function cg(n, tol, it_max)
    it_max = 25
    # find solution and iteration steps
    sol , it = unit_test(n, tol, it_max)
    # print solution
    print("x = ", sol)
    # print number of iteration steps
    print("\n Steps = ", it)
    # drawing plot
    plot(m)
    
