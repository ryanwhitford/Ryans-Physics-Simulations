import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.special as sp
import scipy

# The harmonic oscillator potential is defined to be V(x) = 0.5 * m * omega^2 * x^2
# Note this is the same potential as the classical harmonic oscillator
# It is important because it can be used to approximate more complex potentials at their minima
# This is easy to prove by Taylor expanding the potential around the minimum

# We start by writing the time independent Schrodinger equation
# -hbar^2 / 2m * d^2/dx^2 * psi(x) + 0.5 * m * omega^2 * x^2 * psi(x) = E * psi(x)
# If we discretize space, the Hamiltonian operator becomes a matrix
# The eigenstates can then be found through computational methods

hbar = 1.0546e-34
n=1000 # discretize the x-axis into n points
a=-0.5
b=0.5
x = np.linspace(a, b, n)
dx = (b-a)/n

# Define the potential
omega = 1/10
m = 9.1094e-31
V = 0.5 * m * omega**2 * x**2
V_matrix = np.diag(V)

main_diag = np.ones(n) * -2 # Main diagonal (e.g., 2s on the main diagonal)
off_diag = np.ones(n-1) * 1 # Off diagonals (e.g., -1s on the off-diagonals)
tridiag_matrix = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

H = -(hbar**2 / (2 * m * dx**2)) * tridiag_matrix + V_matrix

#set first and last column to zero to deal with boundary conditions
H[:,0] = 0
H[:,-1] = 0

eigenvalues, eigenvectors = linalg.eigh(H)

def psiNumerical(n):
    return eigenvectors[:,n] / np.sqrt(np.trapz(eigenvectors[:,n]**2, x))

# define function with analytical solution
def psiAnalytic(n, x):
    return (1 / np.sqrt(2**n * np.math.factorial(n)))* (m*omega/(np.pi * hbar))**0.25 * np.exp(-m*omega*x**2 / (2*hbar)) * sp.eval_hermite(n, np.sqrt(m*omega/hbar) * x)

#plot the first four states to check they're equal
n_values = [0,1,2,3]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid of subplots

for i, N in enumerate(n_values):
    row = i // 2  # Determine the row (0 or 1)
    col = i % 2   # Determine the column (0 or 1)
    
    axes[row, col].plot(x, psiAnalytic(N, x), label=f'Analytic')
    axes[row, col].plot(x, psiNumerical(N+2), label=f'Numerical')
    axes[row, col].set_xlabel('x')
    axes[row, col].set_ylabel(r'$\psi(x)$')
    axes[row, col].set_title(f'Eigenstate n={N}')
    axes[row, col].legend()
    axes[row, col].grid(True)
    
fig.suptitle('Wavefunctions for the first 4 eigenstates of the Quantum Harmonic Oscillator')
plt.show()

#find the probablity densities by squaring the wavefunctions

probDensityAnalytic = lambda n, x: psiAnalytic(n, x)**2
probDensityNumerical = lambda n: psiNumerical(n)**2

#Plot the probability densities for the first four states
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid of subplots

for i, N in enumerate(n_values):
    row = i // 2  # Determine the row (0 or 1)
    col = i % 2   # Determine the column (0 or 1)
    
    axes[row, col].plot(x, probDensityAnalytic(N, x), label=f'Analytic')
    axes[row, col].plot(x, probDensityNumerical(N+2), label=f'Numerical')
    axes[row, col].set_xlabel('x')
    axes[row, col].set_ylabel(r'$\psi^2(x)$')
    axes[row, col].set_title(f'Eigenstate n={N}')
    axes[row, col].legend()
    axes[row, col].grid(True)

fig.suptitle('Probability Densities for the first 4 eigenstates of the Quantum Harmonic Oscillator')
plt.show()
