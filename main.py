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
    n = len(A)
    L = np.zeros((n, n))
    for k in range(n):
        L[k, k] = (A[k, k] - sum((L[k, s]**2) for s in range(k)))**0.5
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - sum((L[i, s] * L[k, s]) for s in range(k))) / L[k, k]
    # Input:
    #    A: nxn symmetric, positive definite matrix
    # Output:
    #    L: nxn lower triangular matrix
            
    return L

#####################################################################################################

def PLU(A):
    
    # Input:
    #    A: nxn matrix
    # Output: 
    #    L: nxn lower triangular matrix with unit diagonal
    #    U: nxn upper triangular matrix
    #    P: nxn permutation matrix
    
    n = A.shape[0]
        
    # Initialize vector p
    p = np.arange(n)
    
    # Initialize vector s
    s = np.zeros(n)
    
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for l in range(n):
        s[l] = np.max(np.abs([A[l, :]])) # storing max value from each row
    

    for k in range(n-1):
        
        #Find the pivot element -- do this in one line
        # (Hint: The numpy.argmax function will help)
        PE = np.argmax(np.abs(A[p[k:n], k]) / s[p[k:n]]) + k
        # print("PE = ", PE)
        if p[PE] != p[k]:
            # Swap the pivot element
            p[[k, PE]] = p[[PE, k]]
            
             # Compute scaling factors, and storing it in the values eliminated from
            A[p[k+1:n], k] = A[p[k+1: n], k]/A[p[k], k]
                        
            # using the outer function of numpy as displayed in the stack overflow
        
            A[p[k+1:n], k+1:n] = A[p[k+1:n], k+1:n] - np.outer(A[p[k+1:n], k], A[p[k], k+1:n])
            
    print(A)
            
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A[p[i], j]
            elif i == j:
                L[i, j] = 1
                U[i, j] = A[p[i], j]
            else:
                U[i, j] = A[p[i], j]

    P = np.eye(n)[p]
        
    return L, U, P

#####################################################################################################

def PLUSolve(A,b):
    
    # Input:
    #    A: nxn matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
    
    
    L, U, P = PLU(A)
    n = len(A)
    Pb = P @ b
    x = b
    
    #print("Pb = ", Pb)
    z = np.zeros(n)
    for i in range(n):
        z[i] = Pb[i] - np.dot(L[i, :i], z[:i])
    
    # backword substituion Ux = z
    x = np.zeros(n)
    for i in reversed(range(n)):
        sumOfVal = 0
        for j in range(i + 1, n):
            sumOfVal += U[i, j] * x[j]
        x[i] = (z[i] - sumOfVal)/U[i][i]

        
    return x

#####################################################################################################

def vector_2norm(x):
    return np.sqrt(np.sum(x**2))

    

#####################################################################################################

def matrix_inf_norm(A):
    return np.max(np.sum(np.abs(A), axis=1))
 
    

#####################################################################################################

def inner_product(x, y):
    return np.sum(x * y)

#####################################################################################################

def NeumannSeries(A, tol = 1e-12):
    if matrix_inf_norm(A) >= 1:
        return "B = (I - A)^-1 cannot be computed as matrix_inf_norm(A) >= 1"
    
    B = np.eye(A.shape[0])
    
    # starting at the identity matrix
    newArr = np.eye(A.shape[0]) 
    while (matrix_inf_norm(newArr) > tol):
        newArr = np.dot(newArr, A)
        B += newArr
    
    # we will compute now (I - A) * B
    identity = np.eye(A.shape[0])
    IdentityMinusA = identity - A
    result = np.dot(IdentityMinusA, B)
    
    print(result)
    
    # Input:
    #    A: nxn matrix such that ||A|| < 1
    # Output:
    #    B: nxn inverse (I - A)^{-1}
        
    return B

#####################################################################################################

def Jacobi(A, b):
    # Input:
    #    A: nxn strictly diagonally dominant matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
    D = np.diag(A)
    DMatrix = np.diag(D)
    Dinv = np.linalg.inv(DMatrix)
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = L + U
    
    x = np.zeros_like(b)

    # Initial residual
    r = b - A @ x
    
    # Initialize relative residual vector
    residual = []
    
    tol = 1e-6

    # Implement Jacobi method
    for iteration in range(1000):
        
    
        # Compute Jacobi iteration
        x = Dinv @ (b - (M @ x))
    
        # Compute residual
        r = b - A @ x
        
    
        # Store relative residual in vector
        relative_residual = vector_2norm(r)/vector_2norm(b)
        residual.append(relative_residual)
        
        if relative_residual < tol:
            break
                
    return x

#####################################################################################################

def GaussSeidel(A, b):
    # Input:
    #    A: nxn strictly diagonally dominant matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
    x = np.zeros_like(b)
    n = len(A)
    for k in range(100):
        for i in range(n):
            rowSum = 0
            for j in range(n):
                if i != j:
                    rowSum += A[i][j] * x[j]
            x[i] = (b[i] - rowSum)/A[i][i]
            
        print(f"Iteration {k+1}: {x}")
        
    return x
    

    
#####################################################################################################

def SOR(A, b, w):
    
    # Input:
    #    A: nxn strictly diagonally dominant matrix
    #    b: nx1 vector
    #    w: scalar relaxation parameter
    # Output:
    #    x: nx1 vector
    
    n = len(b)
    x = np.zeros_like(b)
    residuals = []
    
    tol = 1e-6
    
    for iteration in range(1000):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            
            x[i] = (1 - w) * x_old[i] + (w / A[i, i]) * (b[i] - sum1 - sum2)
            
        residual = sum((sum(A[i][j] * x[j] for j in range(n)) - b[i])**2 for i in range(n))**0.5
        residuals.append(residual)
        
        if residual < tol:
            print(f"{iteration + 1} iterations with residual {residual:.2e}")
            break
    

        
    return x

#####################################################################################################

def CG(A, b):
    
    # Input:
    #    A: nxn symmetric positive definite matrix
    #    b: nx1 vector
    # Output:
    #    x: nx1 vector
    
    
    n = len(b)
    x = np.zeros(n)
    
    r = np.array([b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)])
    v = r.copy()
    r_norm = vector_2norm(r)**2
    
    eps = 1e-6
    
    for k in range(1000):
        Av = np.array([sum(A[i][j] * v[j] for j in range(n)) for i in range(n)])
        
        t = r_norm / sum(v[i] * Av[i] for i in range(n))
        
        x = x + t * v
        
        r = r - t * Av
        
        r_norm_new = vector_2norm(r)**2
        
        if vector_2norm(r) < eps:
            print(f"{k + 1} iterations with residual {vector_2norm(r):.2e}")
            break
        
        s = r_norm_new/r_norm
        
        v = r + s * v

        r_norm = r_norm_new
        
        
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

A = np.array([
    [4., 1., 0.],
    [1., 3., -1.],
    [0., -1., 2.]
])

b = np.array([15., 10., 10.])

print(CG(A, b))
# print(L)
# print(U)
# print(P)
    