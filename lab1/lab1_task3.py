"""
@author: mrcraken
@since: 2018
"""

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

if __name__ == '__main__':
    """
    :param n: size of matrix
    """
    n = 4
    M = hilbert_matrix(n)
    try:
        isPD(M)
        B = la.cholesky(hilbert_matrix(4))
        
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.det.html
        Det = la.linalg.det(M)
        
        print("Cholesky: \n" , B)
        print("\n DET of Hilbert matrix = " , Det)
    except la.LinAlgError:
        print("Error")
