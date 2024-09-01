import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

"""
The double pendulum is a classic example of a chaotic system. The trajectory of motion of the system differ greatly with
its initial conditions. The system can be described relatively easily using Lagrangian mechanics. If you would like to see the
derivation I have listed a refererence below. 

Reference:
https://dassencio.org/33
"""

# Define constants
g = 9.81
m1 = 6
m2 = 2
L1 = 3
L2 = 3

# Define sample initial conditions [theta1, theta2, omega1, omega2]
initialConditions_sample = [np.pi/2, np.pi/2, 0, 0]

# Define the time array
a = 0 # start time
b = 20 # end time
N = 30*(b-a)
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

# Function to animate the double pendulum
def animateDoublePendulum(initialConditions=initialConditions_sample, saveas=None):
    # Get the solution for the angles
    theta1Vals, theta2Vals = doublePendulum(initialConditions)
    
    # Convert from polar to cartesian coordinates for the pendulum positions
    x1 = L1 * np.sin(theta1Vals)
    y1 = -L1 * np.cos(theta1Vals)
    x2 = x1 + L2 * np.sin(theta2Vals)
    y2 = y1 - L2 * np.cos(theta2Vals)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1.5 * (L1 + L2), 1.5 * (L1 + L2))
    ax.set_ylim(-1.5 * (L1 + L2), 1.5 * (L1 + L2))
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(r'Double Pendulum with Initial Conditions: $(\theta_1, \theta_2, \omega_1, \omega_2)=$(' + 
                 str(np.round(initialConditions[0], 2)) + ', ' + 
                 str(np.round(initialConditions[1], 2)) + ', ' + 
                 str(np.round(initialConditions[2], 2)) + ', ' + 
                 str(np.round(initialConditions[3], 2)) + ')')

    # Initialize the lines for the two pendulums and the trace
    line1, = ax.plot([], [], 'o-', lw=2, label='Pendulum 1')
    line2, = ax.plot([], [], 'o-', lw=2, label='Pendulum 2')
    trace, = ax.plot([], [], 'r-', lw=1, label='Trace of Pendulum 2')
    
    # List to store the trace points of the second pendulum's tip
    x2_trace = []
    y2_trace = []

    # Initialization function for the animation
    def init():
        # Clear the trace lists at the start of each animation cycle
        x2_trace.clear()
        y2_trace.clear()
        line1.set_data([], [])
        line2.set_data([], [])
        trace.set_data([], [])
        return line1, line2, trace,

    # Update function for the animation
    def update(i):
        # Pendulum 1 
        line1.set_data([0, x1[i]], [0, y1[i]])
        # Pendulum 2
        line2.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        # Append the current tip of pendulum 2 to the trace
        x2_trace.append(x2[i])
        y2_trace.append(y2[i])
        trace.set_data(x2_trace, y2_trace)
        return line1, line2, trace,
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=20)
    
    # Save the animation as a .mp4 file
    if saveas:
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(saveas, writer=writer)
    
    # Display the animation
    plt.legend()
    plt.show()   
    return anim


# Example usage
# Try different initial conditions to see the different trajectories!
# input: [theta1, theta2, omega1, omega2]
# where theta1 and theta2 are the initial angles of the two pendulums and omega1 and omega2 are the initial angular velocities
animateDoublePendulum(initialConditions_sample, 'double_pendulum.mp4')
#animateDoublePendulum([np.pi/2, np.pi, 0, 0])
