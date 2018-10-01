from numpy import linalg as la

def hilbert_matrix(n):
    """
    https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array-in-python
    """
    Matrix = [[1/(i+j-1) for i in range(1,n+1)] for j in range(1,n+1)] 
    return Matrix

def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

if __name__ == '__main__':
    n = 4
    M = hilbert_matrix(n)
    try:
        isPD(M)
        B = la.cholesky(hilbert_matrix(4))
        Det = la.linalg.det(M)
        print("Cholesky: \n" , B)
        print("\n DET of Hilbert matrix = " , Det)
    except la.LinAlgError:
        print("Error")
