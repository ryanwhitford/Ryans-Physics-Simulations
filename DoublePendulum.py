import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
The double pendulum is a classic example of a chaotic system. The trajectory of motion of the system differ greatly with
its initial conditions. The system can be described relatively easily using Lagrangian mechanics. If you would like to see the
derivation I have listed a refererence below. 

Reference:
https://dassencio.org/33
"""

# Define constants
g = 9.81
m1 = 1
m2 = 1
L1 = 1
L2 = 1

# Define sample initial conditions [theta1, theta2, omega1, omega2]
initialConditions_sample = [np.pi/2, np.pi/2, 0, 0]

# Define the time array
N = 1000 # number of points
a = 0 # start time
b = 10 # end time
time = np.linspace(a, b, N)

# Define a function that inputs the current state of the system and returns derivatives of the state
def systemOfODEs(t, state):
    theta1, theta2, omega1, omega2 = state
    f1 = -L2/L1 * (m2 / (m1 + m2)) * omega2**2 * np.sin(theta1 - theta2) - (g / L1) * np.sin(theta1)
    f2 = L1/L2 * omega1**2 * np.sin(theta1 - theta2) - (g / L2) * np.sin(theta2)
    alpha1 = L2/L1 *(m2 / (m1 + m2)) * np.cos(theta1 - theta2)
    alpha2 = L1/L2 * np.cos(theta1 - theta2)
    g1 = (f1 - alpha1 * f2) / (1 - alpha1 * alpha2)
    g2 = (f2 - alpha2 * f1) / (1 - alpha1 * alpha2)
    return [omega1, omega2, g1, g2]

def doublePendulum(initialConditions):
    solution = solve_ivp(systemOfODEs, [a,b], initialConditions, t_eval=time)
    theta1Vals = solution.y[0]
    theta2Vals = solution.y[1]
    return [theta1Vals, theta2Vals]

# Plot trajectory of each pendulum
def plotDoublePendulum(initialConditions = initialConditions_sample):
    # Convert from polar to cartesian coordinates
    x1 = L1 * np.sin(doublePendulum(initialConditions)[0])
    x2 = x1 + L2 * np.sin(doublePendulum(initialConditions)[1])
    y1 = -L1 * np.cos(doublePendulum(initialConditions)[0])
    y2 = y1 - L2 * np.cos(doublePendulum(initialConditions)[1])

    plt.plot(x1, y1, label='Pendulum 1')
    plt.plot(x2, y2, label='Pendulum 2')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Double Pendulum with Initial Conditions: $[\theta 1, \theta 2, \omega 1, \omega 2]=$' + str(np.round(initialConditions,2)))
    plt.legend()
    plt.show()
    return 0

# plot with the defualt initial conditions
#plotDoublePendulum()

# Try different initial conditions! (uncomment the line below)
plotDoublePendulum([np.pi/2, np.pi, 0, 0])
    


