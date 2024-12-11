import numpy as np
import copy

# Final Exam (Part I)
# MATH 3043
# Instructor: Zachary Miksis
# Fall 2024

# Directions: Write 1-2 sentences describing what each function does, including
# any requirements of the method. Then complete each function. All functions should
# be written efficiently (i.e., vectorized or using matrix implementation where 
# applicable). When all functions are complete, run the test notebook as written.
# Do NOT use any numpy linalg functions when completing this library.

#     InvertL(L): 

#     Cholesky(A): 

#     PLU(A): 

#     PLUSolve(A, b): 

#     vector_2norm(x): 

#     matrix_inf_norm(A): 

#     inner_product(x, y): 

#     NeumannSeries(A, tol = 1e-12): 

#     Jacobi(A, b): 

#     GaussSeidel(A, b): 

#     SOR(A, b, w): 

#     CG(A, b): 

#     IncompleteCholesky(A): 

#     PCG(A, b, P = None): 


#####################################################################################################

def InvertL(L):
    
    
    # Input:
    #    L: nxn lower triangular matrix
    # Output:
    #    LINV: nxn lower triangular matrix
    
    n = len(L)
    LINV = np.zeros((n, n))
    for i in range(n):
        LINV[i, i] = 1 / L[i, i]
        for j in range(i + 1, n):
            LINV[j, i] = -sum(L[j, k] * LINV[k, i] for k in range(i, j)) / L[j, j]
                
    return LINV
    
    
        
#####################################################################################################

def Cholesky(A):
    
    # Input:
    #    A: nxn symmetric, positive definite matrix
    # Output:
    #    L: nxn lower triangular matrix
            
    return L

#####################################################################################################

def PLU(A):
    L, U, P = 0, 0, 0
    
    # Input:
    #    A: nxn matrix
    # Output: 
    #    L: nxn lower triangular matrix with unit diagonal
    #    U: nxn upper triangular matrix
    #    P: nxn permutation matrix
    
    return L, U, P

#####################################################################################################

def PLUSolve(A,b):
    x = 0
    
    # Input:
    #    A: nxn matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
        
    return x

#####################################################################################################

def vector_2norm(x):
    return
    

#####################################################################################################

def matrix_inf_norm(A):
    return 
    

#####################################################################################################

def inner_product(x, y):
    return 

#####################################################################################################

def NeumannSeries(A, tol = 1e-12):
    return 
    
    # Input:
    #    A: nxn matrix such that ||A|| < 1
    # Output:
    #    B: nxn inverse (I - A)^{-1}
        
    return B

#####################################################################################################

def Jacobi(A, b):
    x = 0

    # Input:
    #    A: nxn strictly diagonally dominant matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
        
    return x

#####################################################################################################

def GaussSeidel(A, b):
    x = 0
    
    # Input:
    #    A: nxn strictly diagonally dominant matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
    
    return x

#####################################################################################################

def SOR(A, b, w):
    x = 0
    
    # Input:
    #    A: nxn strictly diagonally dominant matrix
    #    b: nx1 vector
    #    w: scalar relaxation parameter
    # Output:
    #    x: nx1 vector
        
    return x

#####################################################################################################

def CG(A, b):
    x = 0
    
    # Input:
    #    A: nxn symmetric positive definite matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
    
    return x

#####################################################################################################

def IncompleteCholesky(A):
    x = 0
    
    # Input:
    #    A: nxn sparse symmetric, positive definite matrix
    # Output:
    #    L: nxn lower triangular matrix
    
    return L

#####################################################################################################

def PCG(A, b, P = None):
    
    x = 0
    
    # Input:
    #    A: nxn symmetric positive definite matrix
    #    b: nx1 vector
    # Optional input:
    #    P: Inverse of preconditioner, P^{-1}
    # Output:
    #    x: nx1 vector
    
    # If no preconditioner, use incomplete Cholesky factorization

    return x


arr = np.array([[2, 0, 0],
               [3, 1, 0],
               [4, 5, 2]])

# arr = np.array([[1, 0],
#                 [3, 1]])


print('Inverted L:')
print(InvertL(arr))
    