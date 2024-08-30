import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import assoc_laguerre as laguerre
from scipy.special import lpmn

"""
The purpose of this demo is to demonstrate numerical methods of solving the 
(time-independent) Schrodinger equation with a well known example: the Hydrogen atom.
Numerical methods are used to solve the Schrodinger equation for much more complex systems
to where analytic solutions are much less obvious. The Hydrogen atom is complex enough to 
motivate the use of numerical methods, even if analytic solutions are available. Because we
are looking at a central potential, I will only be solving for the radial wavefunction. The
angular wavefunctions are the well known spherical harmonics, which i will not solve numerically. 

In the second part of this demo, I will create a 3D plot to illustrate the orbitals of the hydrogen

References:
https://physics.stackexchange.com/questions/170546/numerical-solution-to-schr%C3%B6dinger-equation-eigenvalues
Introduction to Quantum Mechanics 3rd Edition by David J. Griffiths
"""

# Define constants
hbar = 1.05e-34
me = 9.11e-31
e = 1.6e-19
eps0 = 8.85e-12
a0 = 5.29e-11

# Discretize space by creating radius array
N = 500
a = 0 # Lower bound of radius domain, should be zero because we want to see behavior at the nucleus
b = 0.5e-9 # upper bound, can play around with this value as necessary
dr = (b-a)/N # distance between each point in the radius array
r = np.linspace(a+dr, b+dr, N)

# Define the effective potential energy function (V(r) + l(l+1)hbar^2/(2mr^2))
def H(l):
    V = -e**2/(4*np.pi*eps0*r) + l*(l+1)/r**2 * hbar**2 / (2 * me)
    T = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
    T = -hbar**2 * T / (2 * me * dr**2)
    H = T + np.diag(V)
    H[:,-1] = 0
    return H

def RWavefunctionNumerical(n,l):
    if n <= l or n < 1 or l < 0:
        return
    eigenvalues, eigenvectors = linalg.eigh(H(l))
    print(eigenvalues)
    R = eigenvectors[:,n-l] / r # We have been solving for u(r) = r * psi(r). This made the diff eq into the form of 1D schrodinger
    return R / np.sqrt(np.trapz(r**2 * R**2, r)) # Normalize the wavefunction

def RWavefunctionAnalytic(n,l):
    if n <= l or n < 1 or l < 0:
        return
    rho = r/(a0*n)
    def v(p):
        return laguerre(2*p, n-l-1, 2*l + 1)
    R = 1/r * rho**(l+1) * np.exp(-rho) * v(rho)
    return R / np.sqrt(np.trapz(r**2 * R**2, r))

def plotWavefunction(n,l):
    if n <= l or n < 1 or l < 0:
        return
    plt.plot(r, RWavefunctionNumerical(n,l), label='Numerical')
    plt.plot(r, RWavefunctionAnalytic(n,l), label='Analytic')
    plt.legend()
    plt.show()
    return

# The numerical solution should be the same as the analytic solution. Can check by plotting for various n and l.
# Recall the restrictions that n > l , n > 0 , and l >= 0
#plotWavefunction(3,2)

# In three dimensions, the radial wave function is only a portion of the story. There are still angular parts to consider.
# The angular part of the wave function is given by the spherical harmonics
theta = np.linspace(0, np.pi, N)
dtheta = np.pi / N
phi = np.linspace(0, 2*np.pi, N)
dphi = 2*np.pi / N

def ColateralWavefunction(l,m):
    if m < -l or m > l:
        return
    wavefunction = np.zeros(N)
    for i in range(N):
        P_lm, _ = lpmn(m, l, np.cos(theta[i]))  # lpmn returns a tuple, we only need the first part (P_lm)
        wavefunction[i] = P_lm[m, l]  # Extract the specific P_l^m value
    return wavefunction

def AzimuthalWavefunction(m): #m is the magnetic quantum number and can be any integer from -l to l (inclusive)
    return np.exp(1j * m * phi) # note, m being an integer comes from the periodicity condition

def SphericalHarmonic(l,m):
    if m < -l or m > l:
        return
    THETA = ColateralWavefunction(l,m)
    PHI = AzimuthalWavefunction(m)
    normFactor = np.sqrt(np.trapz(np.trapz(np.sin(theta) * np.abs(THETA)**2, theta)*np.abs(PHI)**2, phi))
    return AzimuthalWavefunction(m) / normFactor, ColateralWavefunction(l,m) / normFactor


# I will create a 3D plot that shows all the points where it is probable to find the electron
# Mathematically, this would be where |psi|^2 > a threshold value
threshold = 0.03e27
# I recommend keeping N small for this part of the demo. The 3D plot is very computationally expensive. 
# You can get the general shape of the orbitals with as low as N=10, but I recommend 20-50 for a more accurate representation
def plotOrbital(n, l, m):
    if n <= l or n < l or l < 0 or m < -l or m > l:
        return
    R = RWavefunctionAnalytic(n,l)
    PHI = SphericalHarmonic(l,m)[0]
    THETA = SphericalHarmonic(l,m)[1]
    PSISQUARED = np.outer(R**2, THETA**2) * PHI[0]**2 #|PHI| is constant so no need to create extra dimension
    x = np.zeros(N**3)
    y = np.zeros(N**3)
    z = np.zeros(N**3)
    loopCounter = 0
    integral = 0
    for i in range(N):
        for j in range(N):
            print(np.abs(PSISQUARED[i,j]))
            if np.abs(PSISQUARED[i,j]) > threshold:
                print(np.abs(PSISQUARED[i,j]))
                for k in range(round(3*N/4)):
                    x[loopCounter] = r[i] * np.sin(theta[j]) * np.cos(phi[k])
                    y[loopCounter] = r[i] * np.sin(theta[j]) * np.sin(phi[k])
                    z[loopCounter] = r[i] * np.cos(theta[j])
                    loopCounter += 1   
    #print(np.trapz(np.trapz(np.trapz(R**2 * r**2, r)*np.sin(theta), theta), phi))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    max_range = max(max(x) - min(x), max(y) - min(y), max(z) - min(z))
    mid_x = (max(x) + min(x)) / 2
    mid_y = (max(y) + min(y)) / 2
    mid_z = (max(z) + min(z)) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    plt.show()
    return
"""
This graph is great at visualizing the orbitals of the hydrogen atom, but it is not perfect. The model
requires very fast computer hardware to interact in real time even with N values that give low resolution.
The threshold value is also arbitrary and it doesn't give you any information on the strength of the probability
density, only that it is "probable" to appear in the given area. To fix all of these problems we can visualize 
in another way. I will utilize the fact that probability density is constant with respect to the azimuthal angle.
This means I can plot the probability density as a function of r and theta. I will plot this as a 2D heatmap.
"""
def plotOrbitalHeatmap(n,l,m):
    if n<=l or n<1 or l<0 or m<-l or m>l:
        return
    R = RWavefunctionAnalytic(n,l)
    THETA = SphericalHarmonic(l,m)[1]
    PHI = SphericalHarmonic(l,m)[0]
    PSISQUARED = np.outer(np.abs(THETA)**2, np.abs(R)**2) * np.abs(PHI[0])**2
    
    thetafull = np.concatenate((-theta[::-1], theta))
    PSISQUAREDfull = np.tile(PSISQUARED, (2,1))
    rmesh, thetamesh = np.meshgrid(r, thetafull)
    
    plt.figure(figsize = (8,6))
    ax = plt.subplot(111, polar=True)
    c = ax.pcolormesh(thetamesh,rmesh,PSISQUAREDfull,shading='auto', cmap = 'inferno')
    plt.colorbar(c, ax=ax, label=r'$|\psi(r,\theta,\phi)|^2$') 
    ax.set_title('Probability density of hydrogen with quantum numbers ('+ str(n) + ',' + str(l) + ',' + str(m)+')')
    plt.show()
    
plotOrbitalHeatmap(2,1,0)

    
    