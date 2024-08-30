import numpy as np
import matplotlib.pyplot as plt

# Defining variables
a = 1 # width of the well
x = np.arange(-1,a+1,a/100) # x values

# The infinite well potential is defined as:
# V(x) = 0 for 0 < x < a
# V(x) = infinity for x < 0 and x > a
# The wavefunction cannot exist in infinite potential, so it is 0 outside the well
# Inside the well, the normalized wavefunction is found from the Schrodinger Equation
# The total wave function is piecewise defined as:

def psi(n, x):
    return np.piecewise(x, [x < 0, x >= 0, x > a], [0,
        lambda x : np.sqrt(2/a) * np.sin(np.pi * n * x / a), 0])
    
# where n gives different energy eigenstates of the wavefunction
# Lets plot the wavefunction for the first 4 energy eigenstates
n_values = [1,2,3,4]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid of subplots

for i, n in enumerate(n_values):
    row = i // 2  # Determine the row (0 or 1)
    col = i % 2   # Determine the column (0 or 1)
    
    axes[row, col].plot(x, psi(n, x), label=f'n={n}')
    axes[row, col].set_xlabel('x')
    axes[row, col].set_ylabel(r'$\psi(x)$')
    axes[row, col].set_title(f'Eigenstate n={n}')
    axes[row, col].legend()
    axes[row, col].grid(True)
fig.suptitle('Wavefunctions for the first 4 eigenstates of the Infinite Well')

# The wavefunction still doesn't tell the whole story, we need to find the probability density
# The probability density tells us where the particle is most likely to be found
# Luckily, there is only one more step. The probability density is simply the square of psi

def probDensity(n, x):
    return psi(n, x)**2

# Lets plot the probability density for the first 4 eigenstates
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid of subplots

for i, n in enumerate(n_values):
    row = i // 2  # Determine the row (0 or 1)
    col = i % 2   # Determine the column (0 or 1)
    
    axes[row, col].plot(x, probDensity(n, x), label=f'n={n}')
    axes[row, col].set_xlabel('x')
    axes[row, col].set_ylabel(r'$\psi(x)$')
    axes[row, col].set_title(f'Eigenstate n={n}')
    axes[row, col].legend()
    axes[row, col].grid(True)
plt.suptitle('Probability Density for the first 4 eigenstates of the Infinite Well')
plt.show()

# To double check that the probability density is normalized, we can integrate it over all space
# The integral of the probability density should be 1
# Physically, this means the particle must be somewhere in the well
# Lets check this for the first 4 eigenstates

normalizationCounter = 0
for n in n_values:
    integral = round(np.trapz(probDensity(n, x), x),2)
    if integral == 1:
        normalizationCounter += 1
if normalizationCounter == 4:
    print('The probability densities are normalized!')
else:
    print('The probability densities are NOT normalized')
