import numpy as np

#PARAMETERS
N_CARS = 23
RING_LENGTH = 230
V_MAX = 11.17
ALPHA = 20.0 #sensitivity to FtL term
BETA = 0.5 #sensitivity to optimal velocity term
CAR_LENGTH = 3.6
d_0=2.5

def headway(x_self, x_leader):
    return (x_leader - x_self - CAR_LENGTH + RING_LENGTH) % RING_LENGTH

def optimal_velocity(car, h):
    return V_MAX * (np.tanh((h/d_0)-2)+np.tanh(2))/(1+np.tanh(2))


def acceleration(car, x, v, x_leader, v_leader):
    h = headway(x, x_leader)
    FtL_term = ALPHA * (v_leader - v) / (h**2)
    OV_term = BETA * (optimal_velocity(car, h) - v)
    return FtL_term + OV_term

initial_positions = np.linspace(RING_LENGTH, 0, N_CARS, endpoint=False) #evenly spaced around the ring, with the first car at the end of the ring
initial_velocities = np.full(N_CARS, 8.371723575430472) #every car starts at ideal velocity
initial_state = np.empty(2 * N_CARS)
for i in range(N_CARS):
    initial_state[2*i] = initial_positions[i]
    initial_state[2*i + 1] = initial_velocities[i]
initial_state[1] -= 0.0 #add a small perturbation to the velocity of the first car to break symmetry


def derivatives(t_notused, state):
    deriv = np.empty_like(state)
    for follower in range(N_CARS):
        leader = (follower - 1) % N_CARS
        x_follower, v_follower = state[2*follower], state[2*follower + 1]
        x_leader, v_leader = state[2*leader], state[2*leader + 1]
        deriv[2*follower] = v_follower
        deriv[2*follower + 1] = acceleration(follower, x_follower, v_follower, x_leader, v_leader)
    return deriv

from scipy.integrate import solve_ivp
t_span = (0, 500)
solution = solve_ivp(derivatives, t_span, initial_state, method="Radau", t_eval=np.linspace(0, 500, 3000), atol=1e-9, rtol=1e-9)

import matplotlib.pyplot as plt
for i in range(N_CARS):
    plt.plot(solution.t, solution.y[2*i], label=f'Car {i}')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Bando Ring Simulation')
plt.legend()
plt.show()

for i in range(N_CARS):
    plt.plot(solution.t, solution.y[2*i + 1], label=f'Car {i}')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Bando Ring Simulation - Velocities')
plt.legend()
plt.show()

#headways over time
for i in range(N_CARS):
    leader = (i - 1) % N_CARS
    headways = (solution.y[2*leader] - solution.y[2*i] -CAR_LENGTH + RING_LENGTH) % RING_LENGTH
    plt.plot(solution.t, headways, label=f'Car {i}')
plt.xlabel('Time (s)')
plt.ylabel('Headway (m)')
plt.title('Bando Ring Simulation - Headways')
plt.legend()
plt.show()

#render the positions of the cars on the ring over time as an animation on a ring (so we'll calculate polar coordinates and plot them in a circle)
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

radius = RING_LENGTH / (2 * np.pi)
circle = plt.Circle((0, 0), radius, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
ax.set_xlim(-radius * 1.5, radius * 1.5)
ax.set_ylim(-radius * 1.5, radius * 1.5)

def polar_transform(x):
    angle = (x / RING_LENGTH) * 2 * np.pi
    return radius * np.cos(angle), radius * np.sin(angle)

n_cars = solution.y.shape[0] // 2
colors = ['red'] + ['blue'] * (n_cars - 1)

# Initialize with n_cars dummy points instead of empty arrays
scat = ax.scatter([0] * n_cars, [0] * n_cars, c=colors, s=64)

def update(frame):
    x_positions = solution.y[0::2, frame]
    xs, ys = zip(*[polar_transform(x) for x in x_positions])
    scat.set_offsets(np.column_stack([xs, ys]))
    return scat,

ani = FuncAnimation(fig, update, frames=solution.y.shape[1], blit=True, interval=20)
plt.show()