import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.sparse as sp
import scipy.linalg as la

#################################
# Fluid Function Implementations
#################################

def advect(field, vector_field):
        backtraced_positions = np.clip(coordinates - DT * vector_field, 0.0, DOMAIN_SIZE)
        advected_field = interpolate.interpn(points=(x, y), values=field, xi=backtraced_positions)

        return advected_field

def gradient(field):
    dp_dx = np.zeros_like(field)
    dp_dx[1:-1, 1:-1] = (field[2:, 1:-1] - field[0:-2, 1:-1]) / (2.0 * dx)
    dp_dy = np.zeros_like(field)
    dp_dy[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, 0:-2]) / (2.0 * dx)
    gradient_field = np.stack([dp_dx, dp_dy], axis=-1)

    return gradient_field
    
def project(field):
     divergence = np.zeros_like(field[..., 0])
     divergence[1:-1, 1:-1] = (field[2:, 1:-1, 0] - field[0:-2, 1:-1, 0] + field[1:-1, 2:, 1] - field[1:-1, 0:-2, 1]) / (2.0 * dx)

     main = [-2 for i in range(N_POINTS)]
     upper = [1 for i in range(N_POINTS - 1)]
     lower = upper
     B = sp.diags([main, upper, lower], offsets=[0, 1, -1])
     projectionMatrix = sp.kronsum(B, B, format = 'csc')

     pressure = sp.linalg.cg(projectionMatrix, dx**2 * divergence.flatten(), maxiter=None)[0].reshape((N_POINTS, N_POINTS))
     return (field - gradient(pressure))

def diffuse(field, diffuse_factor):
    A = np.zeros((2*N_POINTS + 1, N_POINTS ** 2))
    mu = diffuse_factor * DT / (dx ** 2)
    A[N_POINTS, :] = 1 + 4 * mu
    A[N_POINTS-1, 1:] = -mu
    A[N_POINTS+1, 1:] = -mu
    A[0, N_POINTS:] = -mu
    A[2*N_POINTS, N_POINTS:] = -mu

    u = field.flatten()
    u = la.solve_banded((N_POINTS, N_POINTS), A, u)
    return np.reshape(u, (N_POINTS, N_POINTS))


############################
# Simulate Specific Problem
############################

def forcing_function(time, point):
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)
    force_location = np.where((point[0] > 0.4) & (point[0] < 0.6) & (point[1] > 0.1) & (point[1] < 0.3), np.array([0.0, 1.0]), np.array([0.0, 0.0]))

    return force_location * time_decay

forcing_function_vectorized = np.vectorize(pyfunc=forcing_function, signature="(),(d)->(d)")

# Define problem size and fluid properties
DOMAIN_SIZE = 1.0
N_POINTS = 41
DT = 0.05
VISCOSITY = 0.0000001        

# Calculate coordinates to be used
dx = DOMAIN_SIZE / (N_POINTS - 1)
scalar_field_shape = (N_POINTS, N_POINTS)
vector_field_shape = (N_POINTS, N_POINTS, 2)
x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

X, Y = np.meshgrid(x, y, indexing="ij")
coordinates = np.stack((X, Y), axis=-1)

# Define inital velocity and substance field
velocities_prev = np.zeros(vector_field_shape)
substance_prev = np.zeros(scalar_field_shape)
substance_prev[10:-10, 10:-10] = 1.0

plt.style.use("dark_background")
plt.figure(figsize=(5, 5), dpi=160)

total_time = 0.0

while (True):
    total_time += DT

    velocities_force_applied = velocities_prev + forcing_function_vectorized(total_time, coordinates) * DT
    velocities_advected = advect(velocities_force_applied, velocities_force_applied)

    velocities_diffused = np.zeros_like(velocities_advected)
    velocities_diffused[..., 0] = diffuse(velocities_advected[..., 0], VISCOSITY)
    velocities_diffused[..., 1] = diffuse(velocities_advected[..., 1], VISCOSITY)

    velocities_projected = project(velocities_advected)
    velocities_prev = velocities_projected

    substance_prev = advect(substance_prev, velocities_projected)

    plt.contourf(X, Y, substance_prev, levels = 100)
    plt.quiver(X, Y, velocities_prev[..., 0], velocities_prev[..., 1], units = 'xy', scale = 5.0, color = 'dimgray')
    plt.text(DOMAIN_SIZE, DOMAIN_SIZE, r'%f' %total_time)

    plt.draw()
    plt.pause(0.0001)
    plt.clf()