import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter

"""
In this simulation I animated the 3 body problem for a system of 3 masses in 2D space.
The equations of motion for the three-body problem are a system of 6 differential equations. (2 spatial dimensions * 3 bodies)
The solutions to these equations are very sensitive to differences in initial conditions. 
This is known as the "butterfly effect" and is a hallmark of chaotic systems.

For my first set of initial conditions, I use values of a stable orbit that I found online from this paper: 
"More than six hundred new families of Newtonian periodic planar collisionless three-body orbits" by Xiaoming Li and Shijun Liao
https://arxiv.org/pdf/1705.00527v4

The second set of initial conditions is a slight perturbation of the first set of initial conditions, where I only changed 
the velocity of the first mass to be 5% greater. The two sets of initial conditions are very similar, but the second set
of initial conditions will lead to a very different trajectory due to the chaotic nature of the three-body problem. This is
why the three body problem is known as a "problem." The problem isn't that we understand the equations that govern planetary
motion, we do and can find numerical solutions of very accurate trajectories as long as we have the exact initial conditions.
The problem lies in the fact that we can never know the exact initial conditions of a system of 3 or more bodies in space. 
Our measurements are always limited by the precision of our instruments, and the smallest error in initial conditions can lead
to vasly different predictions. The equations are useless if they cannot give even approximations of the future state of the 
system. This is why the three body problem is a problem.
"""

# define constants
G = 1
m1 = 1
m2 = m3 = m1

# define sample initial conditions
initialConditions_sample1 = [-1.0024277970, 0.0041695061, 1.0024277970, -0.0041695061, 0, 0, 0.3489048974, 0.5306305100, 0.3489048974, 0.5306305100, -2 * 0.3489048974, -2 * 0.5306305100]
initialConditions_sample2 = [-1.0024277970, 0.0041695061, 1.0024277970, -0.0041695061, 0, 0, 1.05*0.3489048974, 1.05*0.5306305100, 0.3489048974, 0.5306305100, -2 * 0.3489048974, -2 * 0.5306305100]

def equationsOfMotion(t, state):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
    
    # calculate the distance between the bodies
    modr12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    modr13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
    modr23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    
    # find equations of motion (Newton's second law)
    d2_x1_dt2 = - G * m2 * (x1 - x2) / modr12**3 - G * m3 * (x1 - x3) / modr13**3
    d2_y1_dt2 = - G * m2 * (y1 - y2) / modr12**3 - G * m3 * (y1 - y3) / modr13**3
    d2_x2_dt2 = - G * m3 * (x2 - x3) / modr23**3 - G * m1 * (x2 - x1) / modr12**3
    d2_y2_dt2 = - G * m3 * (y2 - y3) / modr23**3 - G * m1 * (y2 - y1) / modr12**3
    d2_x3_dt2 = - G * m1 * (x3 - x1) / modr13**3 - G * m2 * (x3 - x2) / modr23**3
    d2_y3_dt2 = - G * m1 * (y3 - y1) / modr13**3 - G * m2 * (y3 - y2) / modr23**3
    
    return [vx1, vy1, vx2, vy2, vx3, vy3, d2_x1_dt2, d2_y1_dt2, d2_x2_dt2, d2_y2_dt2, d2_x3_dt2, d2_y3_dt2]

def threeBodyProblem(initialConditions, end_time, fps):
    # create the time array
    t = np.linspace(0, end_time, fps*end_time)
    solution = solve_ivp(equationsOfMotion, [0, end_time], initialConditions, t_eval=t)
    
    x1_vals = solution.y[0]
    y1_vals = solution.y[1]
    x2_vals = solution.y[2]
    y2_vals = solution.y[3]
    x3_vals = solution.y[4]
    y3_vals = solution.y[5]
    
    return [x1_vals, y1_vals, x2_vals, y2_vals, x3_vals, y3_vals]
    
def animateThreeBodyProblem(initialConditions=initialConditions_sample1, end_time=20, fps=60, initialConditions2=None, saveas=None):
    # Get the solution of the three-body problem for the first initial conditions
    x1_vals, y1_vals, x2_vals, y2_vals, x3_vals, y3_vals = threeBodyProblem(initialConditions, end_time=end_time, fps=fps)
    
    # If initialConditions2 is provided, solve the second set of initial conditions
    if initialConditions2 is not None:
        x1_vals2, y1_vals2, x2_vals2, y2_vals2, x3_vals2, y3_vals2 = threeBodyProblem(initialConditions2, end_time=end_time, fps=fps)
    
    max_x = max(max(x1_vals), max(x2_vals), max(x3_vals))
    min_x = min(min(x1_vals), min(x2_vals), min(x3_vals))
    min_y = min(min(y1_vals), min(y2_vals), min(y3_vals))
    max_y = max(max(y1_vals), max(y2_vals), max(y3_vals))
    
    if initialConditions2 is not None:
        max_x2 = max(max(x1_vals2), max(x2_vals2), max(x3_vals2))
        min_x2 = min(min(x1_vals2), min(x2_vals2), min(x3_vals2))
        min_y2 = min(min(y1_vals2), min(y2_vals2), min(y3_vals2))
        max_y2 = max(max(y1_vals2), max(y2_vals2), max(y3_vals2))
    else:
        max_x2 = max_x
        min_x2 = min_x
        min_y2 = min_y
        max_y2 = max_y
    
    xlim_value = 1.25 * max(abs(max_x), abs(min_x), abs(max_x2), abs(min_x2))
    ylim_value = 1.25 * max(abs(max_y), abs(min_y), abs(max_y2), abs(min_y2))
    
    # Create the figure and axes
    if initialConditions2 is None:
        fig, ax = plt.subplots()
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    for ax in axs:
        ax.set_xlim(-xlim_value, xlim_value)
        ax.set_ylim(-ylim_value, ylim_value)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    axs[0].set_title('Three-Body Problem (Initial Conditions 1)')
    
    if initialConditions2 is not None:
        axs[1].set_title('Three-Body Problem (Initial Conditions 2)')
    
    # Create the lines for the bodies and the trajectories for the first set of initial conditions
    body1, = axs[0].plot([], [], 'ro', markersize=8, label='Mass 1')
    body2, = axs[0].plot([], [], 'bo', markersize=8, label='Mass 2')
    body3, = axs[0].plot([], [], 'go', markersize=8, label='Mass 3')
    
    traj1, = axs[0].plot([], [], 'r-', linewidth=1)  # Trajectory for body 1
    traj2, = axs[0].plot([], [], 'b-', linewidth=1)  # Trajectory for body 2
    traj3, = axs[0].plot([], [], 'g-', linewidth=1)  # Trajectory for body 3
    
    # If initialConditions2 is provided, create the lines for the second set of initial conditions
    if initialConditions2 is not None:
        body1_2, = axs[1].plot([], [], 'ro', markersize=8, label='Mass 1')
        body2_2, = axs[1].plot([], [], 'bo', markersize=8, label='Mass 2')
        body3_2, = axs[1].plot([], [], 'go', markersize=8, label='Mass 3')
        
        traj1_2, = axs[1].plot([], [], 'r-', linewidth=1)
        traj2_2, = axs[1].plot([], [], 'b-', linewidth=1)
        traj3_2, = axs[1].plot([], [], 'g-', linewidth=1)
    
    # Initialize lists to store trajectory data for the first set of initial conditions
    x1_traj, y1_traj = [], []
    x2_traj, y2_traj = [], []
    x3_traj, y3_traj = [], []
    
    # If initialConditions2 is provided, initialize lists for the second set of initial conditions
    if initialConditions2 is not None:
        x1_traj2, y1_traj2 = [], []
        x2_traj2, y2_traj2 = [], []
        x3_traj2, y3_traj2 = [], []
    
    # Initialize the animation
    def init():
        x1_traj.clear()
        y1_traj.clear()
        x2_traj.clear()
        y2_traj.clear()
        x3_traj.clear()
        y3_traj.clear()
        
        body1.set_data([], [])
        body2.set_data([], [])
        body3.set_data([], [])
        traj1.set_data([], [])
        traj2.set_data([], [])
        traj3.set_data([], [])
        
        if initialConditions2 is not None:
            x1_traj2.clear()
            y1_traj2.clear()
            x2_traj2.clear()
            y2_traj2.clear()
            x3_traj2.clear()
            y3_traj2.clear()
            
            body1_2.set_data([], [])
            body2_2.set_data([], [])
            body3_2.set_data([], [])
            traj1_2.set_data([], [])
            traj2_2.set_data([], [])
            traj3_2.set_data([], [])
        
        return body1, body2, body3, traj1, traj2, traj3
    
    # Update function for the animation
    def update(frame):
        # Update body positions for the first set of initial conditions
        body1.set_data([x1_vals[frame]], [y1_vals[frame]])
        body2.set_data([x2_vals[frame]], [y2_vals[frame]])
        body3.set_data([x3_vals[frame]], [y3_vals[frame]])
        
        # Append the current positions to the trajectory lists for the first set of initial conditions
        x1_traj.append(x1_vals[frame])
        y1_traj.append(y1_vals[frame])
        x2_traj.append(x2_vals[frame])
        y2_traj.append(y2_vals[frame])
        x3_traj.append(x3_vals[frame])
        y3_traj.append(y3_vals[frame])
        
        # Update trajectories for the first set of initial conditions
        traj1.set_data(x1_traj, y1_traj)
        traj2.set_data(x2_traj, y2_traj)
        traj3.set_data(x3_traj, y3_traj)
        
        # If initialConditions2 is provided, update the second set of initial conditions
        if initialConditions2 is not None:
            body1_2.set_data([x1_vals2[frame]], [y1_vals2[frame]])
            body2_2.set_data([x2_vals2[frame]], [y2_vals2[frame]])
            body3_2.set_data([x3_vals2[frame]], [y3_vals2[frame]])
            
            x1_traj2.append(x1_vals2[frame])
            y1_traj2.append(y1_vals2[frame])
            x2_traj2.append(x2_vals2[frame])
            y2_traj2.append(y2_vals2[frame])
            x3_traj2.append(x3_vals2[frame])
            y3_traj2.append(y3_vals2[frame])
            
            traj1_2.set_data(x1_traj2, y1_traj2)
            traj2_2.set_data(x2_traj2, y2_traj2)
            traj3_2.set_data(x3_traj2, y3_traj2)
        
        return body1, body2, body3, traj1, traj2, traj3
    
    # Create the animation
    frames = len(x1_vals)  # Number of frames corresponds to the number of data points
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=1000/fps)
    
    # Save the animation if a filename is provided
    if saveas:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(saveas, writer=writer)
    
    plt.show()
    return anim

animateThreeBodyProblem(fps = 60, initialConditions2=initialConditions_sample2, saveas='three_body_problem.mp4')
